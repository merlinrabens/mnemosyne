# Architecture

mnemosyne is two files:

| File | Role | hermes-agent dependency |
|---|---|---|
| `mnemosyne/unified_index.py` | The retrieval **core** — embedder, schema, chunking, upsert, hybrid search. Pure Python + `fastembed` + `sqlite-vec`. | **None** (importable & testable standalone) |
| `mnemosyne/__init__.py` | The **provider** — implements hermes-agent's `MemoryProvider` ABC, wires tools/prefetch/hooks onto the core. | The ABC only |

That split is deliberate: the core is the valuable, reusable part; the
provider is a thin adapter. Anything else (a CLI, an MCP server, a cron
indexer) can import the core and share one retrieval code path.

## One SQLite file, three tables

```
chunks       -- chunk text + metadata (doc_id, source, mtime, lint_flag, …)
chunks_fts   -- FTS5 BM25 keyword index   (chunk_id → content)
chunks_vec   -- sqlite-vec vec0 vector idx (rowid = chunk_id → float[1024])
```

`chunk_id` is the shared key across all three. Every chunk has exactly
one row in each — enforced by writing all three inside a single
`BEGIN IMMEDIATE` transaction per document. A crashed writer never leaves
the keyword index and the vector index disagreeing about what exists.

vec0 has no `UPSERT`, so re-indexing a document is **delete-by-doc_id
across all three tables, then re-insert**. A content-SHA guard
(`doc_sha`) short-circuits the expensive embed step when a document is
unchanged — re-indexing an unchanged corpus is O(docs) cheap SELECTs.

## Hybrid retrieval = one SQL statement, RRF k=60

```
score = w_fts/(60 + fts_rank) + w_vec/(60 + vec_rank)  +  recency
```

Two CTEs produce independent rank lists — BM25 (`bm25(chunks_fts)`) and
vector KNN (`embedding MATCH :q AND k = :pool`) — joined `FULL OUTER` on
`chunk_id` and fused by **Reciprocal Rank Fusion**.

The key property: RRF consumes only the *rank* from each arm, never the
raw score. BM25 magnitudes and L2 distances live on incomparable scales;
RRF never has to normalise them onto a common axis. That scale-invariance
is why RRF reliably beats score-weighted fusion for keyword+vector
hybrids, and it's the published pattern from the sqlite-vec author.

A small **recency tie-breaker** is added in Python after the SQL (stock
SQLite has no `exp()`): `peak · 0.5^(age_days / halflife)`, bounded so
its maximum contribution is ≈30% of a single RRF rank step. It only ever
flips genuine near-ties (a fresh note edging out a stale one on the same
topic) — never enough to surface an irrelevant-but-fresh document.

## Why zero torch

The embedder is `fastembed` (ONNX / onnxruntime), model
`mixedbread-ai/mxbai-embed-large-v1` (dim 1024, MTEB ≈ 64.7, ~0.64 GB).
It never imports `torch`. This is not incidental — it's the central
design constraint:

- `torch ≥ 2.6` has **no x86_64-macOS wheels**; the last Intel-Mac torch
  is 2.2.2, which is NumPy-1.x-ABI and breaks under NumPy 2.x. A
  torch-based local embedder is a permanent fragility trap on that
  platform (and a heavy dependency everywhere else).
- ONNX runtime sidesteps the entire torch/NumPy-ABI problem class.
- The model output is **L2-normalised**, so vec0's default L2 distance
  is monotonic with cosine similarity — no normalisation code needed,
  KNN order == cosine order for free.

`import torch` succeeding anywhere in the embed path is therefore a
regression, not a nicety. CI does not install torch.

## Self-healing dependencies

`unified_index._ensure_deps()` runs lazily before the first embed/connect.
If `fastembed`/`sqlite-vec` are missing (e.g. a host venv was regenerated
by an upgrade), it reinstalls the pinned versions **once** into the
running interpreter, then continues — guarded against loops (one attempt
per process; a 24 h on-disk throttle across processes).

Effect: a venv-regenerating host upgrade degrades to *"the first call
after the upgrade is a few seconds slower"* — not *"semantic memory
silently gone until someone notices"*. Combined with living in the
host's user-plugin directory (outside its venv), the provider survives
host upgrades structurally rather than by luck.

## Concurrency

WAL mode + a 5 s busy-timeout. Many concurrent readers, one writer. The
provider opens a fresh connection per operation because the host calls
`prefetch`/tools from multiple threads and `sqlite3` connections are not
thread-safe. The embedder is a process-wide lazy singleton; a background
keep-warm thread loads it off the critical path so no user-facing call
pays the one-time model cold-load.

## Write safety in multi-context hosts

`initialize(agent_context=…)` gates writes: only a `"primary"` context
may write memory. Cron / subagent / flush contexts are read-only — their
system prompts must never be distilled into "user facts". Reads work in
every context; only the write paths are gated.
