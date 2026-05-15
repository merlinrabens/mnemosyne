<div align="center">

<img src="assets/logo.svg" alt="mnemosyne" width="140" />

# mnemosyne

**Local-first hybrid semantic memory for [hermes-agent](https://github.com/NousResearch/hermes-agent).**
One corpus, one retrieval path: keyword (BM25) ⊕ neural vector, RRF-fused — **zero torch, zero cloud, zero daemon.**

<img src="assets/demo.gif" alt="mnemosyne selftest demo" width="760" />

</div>

---

## What it is

A drop-in `MemoryProvider` for hermes-agent that unifies **stored memory
facts + any document sources** (a wiki, an Obsidian vault, …) into one
SQLite file and serves them through a single hybrid query:

> BM25 keyword **⊕** neural-vector KNN, fused by Reciprocal Rank Fusion,
> with a bounded recency tie-breaker.

Embeddings run locally via [`fastembed`](https://github.com/qdrant/fastembed)
(ONNX) — **no torch, no API key, no data leaving the machine**. The
vector index is [`sqlite-vec`](https://github.com/asg017/sqlite-vec),
living in the *same SQLite file* as the FTS5 index. No vector server, no
Docker, no background daemon.

## Why

Most local-memory stacks pull in `torch`. On Intel macOS that's a dead
end (`torch ≥ 2.6` has no x86_64 wheels; the last one is NumPy-1.x-ABI
and breaks under NumPy 2). Cloud embedding APIs solve the fragility but
send your private corpus off-machine — and some reserve the right to
train on it.

mnemosyne takes the third path: a strong **local ONNX** model
(`mxbai-embed-large-v1`, MTEB ≈ 64.7) + hybrid fusion. The raw-embedding
gap to a top cloud model is small and **further closed by the RRF hybrid**
(BM25 covers the exact-identifier cases where pure vectors are weakest).
You trade a couple of MTEB points for total privacy, offline operation,
zero quota, and structural immunity to the torch/NumPy-ABI trap.

## Features

- **Unified corpus** — memory facts and document sources in one index, one query.
- **Hybrid + recency** — RRF(k=60) keyword⊕vector; fresh content edges out stale on near-ties.
- **Zero torch** — ONNX only; CI fails if torch enters the embed path.
- **Crash-consistent** — one `BEGIN IMMEDIATE` txn per doc across all three tables.
- **Idempotent** — content-SHA guard skips re-embedding unchanged docs.
- **Self-healing deps** — survives a host venv regeneration (auto-reinstalls pinned deps once).
- **Upgrade-safe** — lives in the user-plugin dir, outside the host venv.
- **Write-gated** — only `primary` agent contexts write; cron/subagents are read-only.

## Install (as a hermes-agent plugin)

```bash
git clone https://github.com/merlinrabens/mnemosyne.git
cp -r mnemosyne "$HERMES_HOME/plugins/mnemosyne"
"$HERMES_HOME/hermes-agent/venv/bin/pip" install fastembed==0.8.0 sqlite-vec==0.1.9
```

Then in `config.yaml`:

```yaml
memory:
  provider: mnemosyne
plugins:
  mnemosyne:
    auto_extract: true   # pattern-distil salient facts at session end
```

Restart the gateway. `hermes memory status` should show
`mnemosyne (local) ← active`.

> First call downloads the ONNX model (~0.64 GB) once. A background
> keep-warm thread loads it off the critical path so no user turn pays
> the cold-load.

## Tools exposed

| Tool | Purpose |
|---|---|
| `mem_search(query, sources?, limit?)` | Hybrid recall across all (or selected) sources |
| `mem_add(content)` | Store a durable fact (idempotent, embedded on write) |
| `mem_feedback(doc_id)` | Drop a wrong/outdated fact from the index |

The provider also injects per-turn `prefetch` context automatically and
mirrors the host's built-in memory writes.

## Use the core standalone

`unified_index.py` has **no hermes-agent dependency** — it's a usable
hybrid-retrieval library on its own:

```python
from mnemosyne import unified_index as ui
conn = ui.connect("index.db"); ui.init_schema(conn)
ui.upsert_doc(conn, doc_id="notes:1", source="notes",
              text="# Deploy\nBlue-green rollout; failed health probe rolls back.")
ui.hybrid_search(conn, "how does releasing avoid downtime")
```

```bash
python mnemosyne/unified_index.py selftest   # end-to-end, in-memory
```

## How it works

See [`docs/architecture.md`](docs/architecture.md) — the three-table
schema, the RRF SQL, the zero-torch rationale, the self-heal, and the
write-gating model.

## Tests

```bash
pip install -e ".[dev]"
ruff check mnemosyne tests
pytest -q
```

CI runs the real end-to-end stack (fastembed + sqlite-vec, no mocks).

## Credits & provenance

- Built against the [hermes-agent](https://github.com/NousResearch/hermes-agent)
  `MemoryProvider` interface (Nous Research).
- Provider structure follows the pattern of hermes-agent's bundled
  `holographic` provider (MIT) — independently implemented, not copied.
- [`fastembed`](https://github.com/qdrant/fastembed) (Apache-2.0, Qdrant) ·
  [`sqlite-vec`](https://github.com/asg017/sqlite-vec) (MIT, Alex Garcia) ·
  embedding model [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
- RRF hybrid pattern per the sqlite-vec author's published approach.

Named for **Mnemosyne**, the Greek Titaness of *memory* and mother of
the Muses.

## License

[MIT](LICENSE).
