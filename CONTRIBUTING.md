# Contributing

Thanks for considering a contribution.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The first test run downloads the embedding model (`mxbai-embed-large-v1`,
~0.64 GB ONNX) into the fastembed cache. Subsequent runs are fast.

## Before opening a PR

```bash
ruff check mnemosyne tests
pytest -q
python mnemosyne/unified_index.py selftest   # end-to-end sanity
```

All three must pass. CI runs exactly this.

## Design constraints (please respect)

- **Zero torch.** The embedding path must stay ONNX/onnxruntime only.
  `import torch` succeeding anywhere in the embed path is a regression —
  it reintroduces the exact dependency-fragility this project exists to
  avoid. CI does not install torch; keep it that way.
- **The memory layer must never crash its host.** Broad `except` at
  degradation boundaries is deliberate and annotated `# noqa: BLE001`.
  A failed recall returns empty / logs — it does not raise into the
  agent.
- **`unified_index.py` stays free of any hermes-agent import.** It is the
  shared, embeddable core. Only `mnemosyne/__init__.py` (the provider)
  may import the hermes-agent ABC.
- **One transaction per document.** Upsert is `BEGIN IMMEDIATE` →
  delete-by-doc across all three tables → re-insert. Don't split it; the
  crash-consistency of the FTS/vector/row triple depends on it.
- **Schema changes require a `SCHEMA_VERSION` bump** and a note on
  migration (the `index_meta` guard will otherwise refuse mismatched
  databases, by design).

## Scope

Bug fixes, perf, docs, and additional source adapters are welcome.
Changing the embedding model is a breaking change (vectors are not
comparable across models) — discuss in an issue first.
