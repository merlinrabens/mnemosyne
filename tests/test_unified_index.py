"""End-to-end test of the retrieval core with real fastembed + vec0.

`unified_index.py` is imported directly (not via the `mnemosyne` package),
because the package `__init__` imports hermes-agent's MemoryProvider ABC,
which is intentionally absent in CI. The core is dependency-free of
hermes-agent by design — that separation is what this import proves.
"""
import importlib.util
import pathlib

import pytest

_CORE = pathlib.Path(__file__).resolve().parent.parent / "mnemosyne" / "unified_index.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("unified_index", _CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ui():
    return _load_core()


def test_selftest_passes(ui):
    """The bundled selftest exercises upsert, paraphrased semantic recall,
    source-filter isolation and content-SHA idempotency. Exit 0 == PASS."""
    assert ui._selftest() == 0


def test_paraphrase_recall_uses_vector_arm(ui):
    """A query with ~no lexical overlap with its target must still rank it
    first — i.e. the neural arm, not just BM25, is doing the work."""
    conn = ui.connect(":memory:")
    ui.init_schema(conn)
    ui.upsert_doc(
        conn,
        doc_id="wiki:guides/deploy.md",
        source="wiki",
        text=("# Deployment Guide\nThe service ships via a blue-green "
              "rollout. A health probe gates promotion; a failed probe "
              "rolls the release back automatically."),
        kind="markdown",
    )
    ui.upsert_doc(
        conn,
        doc_id="memory:fact:1",
        source="memory",
        text="The user prefers tabs over spaces and a dark editor theme.",
        kind="fact",
    )
    hits = ui.hybrid_search(conn, "how does releasing avoid downtime", k=3)
    assert hits, "expected at least one hit"
    assert hits[0]["doc_id"] == "wiki:guides/deploy.md"
    conn.close()


def test_source_filter_isolation(ui):
    conn = ui.connect(":memory:")
    ui.init_schema(conn)
    ui.upsert_doc(conn, doc_id="wiki:a", source="wiki",
                  text="alpha bravo charlie", kind="fact")
    ui.upsert_doc(conn, doc_id="memory:b", source="memory",
                  text="alpha delta echo", kind="fact")
    only_mem = ui.hybrid_search(conn, "alpha", sources=["memory"], k=5)
    assert only_mem and all(r["source"] == "memory" for r in only_mem)
    conn.close()


def test_idempotent_upsert_skips(ui):
    conn = ui.connect(":memory:")
    ui.init_schema(conn)
    txt = "# Note\nidempotency check body"
    assert ui.upsert_doc(conn, doc_id="d", source="wiki", text=txt) == "indexed"
    assert ui.upsert_doc(conn, doc_id="d", source="wiki", text=txt) == "skipped"
    conn.close()


def test_delete_doc_purges_all_three_tables(ui):
    conn = ui.connect(":memory:")
    ui.init_schema(conn)
    ui.upsert_doc(conn, doc_id="d", source="wiki", text="some content here")
    assert ui.corpus_stats(conn)["chunks"] == 1
    removed = ui.delete_doc(conn, "d")
    assert removed == 1
    assert ui.corpus_stats(conn)["chunks"] == 0
    # vec + fts rows gone too (no orphans → next search returns nothing)
    assert ui.hybrid_search(conn, "content", k=5) == []
    conn.close()
