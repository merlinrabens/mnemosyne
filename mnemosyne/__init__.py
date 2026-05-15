"""mnemosyne — unified local-first semantic memory provider for hermes-agent.

ONE corpus, ONE retrieval path. Any number of named sources (e.g. stored
memory facts + a wiki + an Obsidian vault) are chunked, embedded
(fastembed ``mxbai-embed-large-v1``, **zero torch**), and stored
side-by-side in a single SQLite file (``chunks`` / ``chunks_fts`` /
``chunks_vec``). Retrieval is one RRF-fused hybrid query (BM25 ⊕ neural
vector) with a bounded recency tie-breaker — implemented once in
``unified_index`` and shared verbatim by anything else that wants it.

Drops into hermes-agent's user-plugin slot (``$HERMES_HOME/plugins/
mnemosyne/``) — outside the venv, so it survives ``hermes`` upgrades.
Local, private, offline, free; no cloud embedding API.

Config (``config.yaml`` → ``plugins.mnemosyne``):
  auto_extract: true   # pattern-distil salient facts at session end
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
except ModuleNotFoundError:
    # hermes-agent not importable (repo inspection / CI / lint). Provide a
    # minimal stand-in so the package imports and the class body parses;
    # at runtime inside hermes-agent the real ABC is used.
    class MemoryProvider:  # type: ignore[no-redef]
        """Stand-in base — replaced by hermes-agent's real ABC at runtime."""

from . import unified_index as ui

logger = logging.getLogger(__name__)


# ── Tool schemas ─────────────────────────────────────────────────────────────
MEM_SEARCH_SCHEMA = {
    "name": "mem_search",
    "description": (
        "Semantic + keyword recall across the UNIFIED knowledge base — "
        "stored memory facts plus any indexed document sources (wiki, "
        "notes/Obsidian, …) — as one hybrid (meaning + exact-term), "
        "freshness-aware query. Use this before answering anything about "
        "the user, their projects, prior decisions, or anything that might "
        "be written down. `sources` defaults to all."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to recall."},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subset of source labels to search (default: all).",
            },
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}

MEM_ADD_SCHEMA = {
    "name": "mem_add",
    "description": (
        "Store a durable fact the user would expect you to remember "
        "(preferences, decisions, people, project facts). Embedded "
        "immediately so it is recallable by meaning. Idempotent — storing "
        "the same fact twice is a no-op."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
        },
        "required": ["content"],
    },
}

MEM_FEEDBACK_SCHEMA = {
    "name": "mem_feedback",
    "description": (
        "Signal that a recalled memory fact was wrong/outdated so it stops "
        "surfacing. Removes the fact from the index."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "doc_id of the bad fact (from mem_search)."},
        },
        "required": ["doc_id"],
    },
}


# ── Config ───────────────────────────────────────────────────────────────────
def _load_plugin_config() -> dict:
    try:
        import yaml
        from hermes_cli.config import cfg_get
        from hermes_constants import get_hermes_home

        cfgp = get_hermes_home() / "config.yaml"
        if not cfgp.exists():
            return {}
        with open(cfgp, encoding="utf-8-sig") as f:
            allc = yaml.safe_load(f) or {}
        return cfg_get(allc, "plugins", "mnemosyne", default={}) or {}
    except Exception:
        return {}


def _fact_doc_id(content: str) -> str:
    """Content-addressed → storing the same fact twice dedupes for free."""
    return "memory:" + hashlib.sha1(content.strip().encode("utf-8", "replace")).hexdigest()[:16]


# ── Provider ─────────────────────────────────────────────────────────────────
class MnemosyneMemoryProvider(MemoryProvider):
    """Unified hybrid retrieval over stored facts + document sources."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._db_path = None
        self._session_id = ""
        self._writable = True  # flipped off for non-primary agent contexts

    # -- required -----------------------------------------------------------
    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        if ui is None:
            logger.warning("mnemosyne unavailable — unified_index import failed")
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home", str(Path("~/.hermes").expanduser()))
        self._db_path = str((Path(hermes_home) / "notes-index.db"))
        ctx = kwargs.get("agent_context", "primary")
        # cron/subagent/flush prompts must NOT pollute the user's memory.
        self._writable = ctx == "primary"
        try:
            conn = ui.connect(self._db_path)
            ui.init_schema(conn)
            conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("mnemosyne schema init failed: %s", e)
        # Keep the ONNX model resident so no user-facing prefetch/mem_search
        # ever pays the ~4s cold-load. One persistent keeper rather than a
        # one-shot warm — the cold-load is paid by this background thread.
        threading.Thread(target=self._keep_warm, daemon=True).start()

    def _keep_warm(self) -> None:
        """Load the embedder ASAP and keep it resident.

        Retries until the model loads (covers a transient first-load
        failure or a still-downloading model), then re-touches every 10
        min as a cheap heartbeat — a no-op while the singleton is resident,
        and a self-heal if it was ever lost.
        """
        while True:
            try:
                ui.get_embedder()
                time.sleep(600)
            except Exception as e:  # noqa: BLE001
                logger.debug("mnemosyne keep-warm retry (%s)", e)
                time.sleep(30)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEM_SEARCH_SCHEMA, MEM_ADD_SCHEMA, MEM_FEEDBACK_SCHEMA]

    # -- retrieval ----------------------------------------------------------
    def _open(self):
        # Fresh connection per op: the gateway calls prefetch/tools from
        # multiple threads and sqlite3 connections are not thread-safe.
        c = ui.connect(self._db_path)
        ui.init_schema(c)
        return c

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if ui is None or not query:
            return ""
        try:
            conn = self._open()
            try:
                hits = ui.hybrid_search(conn, query, k=5)
            finally:
                conn.close()
            if not hits:
                return ""
            lines = []
            for h in hits:
                tag = h["source"][:3]
                where = h.get("rel_path") or h.get("doc_id")
                snippet = " ".join(h["content"].split())[:240]
                lines.append(f"- [{tag}] {where}: {snippet}")
            return "## Memory (unified recall)\n" + "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.debug("mnemosyne prefetch failed: %s", e)
            return ""

    def system_prompt_block(self) -> str:
        if ui is None:
            return ""
        try:
            conn = self._open()
            try:
                st = ui.corpus_stats(conn)
            finally:
                conn.close()
        except Exception:
            st = {"chunks": 0, "by_source": {}}
        n = st.get("chunks", 0)
        by = st.get("by_source", {})
        if n == 0:
            return ("# Memory (mnemosyne)\nUnified semantic memory active "
                    "(empty). Use mem_add for durable facts; mem_search to "
                    "recall across all sources.")
        bits = ", ".join(f"{k} {v}" for k, v in sorted(by.items()))
        return (f"# Memory (mnemosyne)\nUnified hybrid recall active — "
                f"{n} chunks ({bits}). ALWAYS mem_search before answering "
                f"about the user, their projects, or prior decisions. "
                f"mem_add stores durable facts; mem_feedback removes wrong ones.")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Explicit facts only (mem_add / on_memory_write / on_session_end).
        # No per-turn auto-write — keeps the corpus signal, not transcript noise.
        pass

    # -- tools --------------------------------------------------------------
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if ui is None:
            return json.dumps({"error": "mnemosyne unavailable"})
        try:
            if tool_name == "mem_search":
                return self._mem_search(args)
            if tool_name == "mem_add":
                return self._mem_add(args)
            if tool_name == "mem_feedback":
                return self._mem_feedback(args)
            return json.dumps({"error": f"unknown tool {tool_name}"})
        except KeyError as e:
            return json.dumps({"error": f"missing argument: {e}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    def _mem_search(self, args: dict) -> str:
        q = args["query"]
        sources = args.get("sources") or None
        limit = int(args.get("limit", 8))
        conn = self._open()
        try:
            hits = ui.hybrid_search(conn, q, sources=sources, k=limit)
        finally:
            conn.close()
        return json.dumps({
            "results": [
                {
                    "doc_id": h["doc_id"], "source": h["source"],
                    "where": h.get("rel_path") or h["doc_id"],
                    "content": h["content"][:1200],
                    "score": round(h["score"], 5),
                }
                for h in hits
            ],
            "count": len(hits),
        })

    def _mem_add(self, args: dict) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return json.dumps({"error": "empty content"})
        if not self._writable:
            return json.dumps({"status": "skipped", "reason": "non-primary context"})
        did = _fact_doc_id(content)
        conn = self._open()
        try:
            r = ui.upsert_doc(conn, doc_id=did, source="memory", text=content,
                              title=None, mtime=time.time(), kind="fact")
        finally:
            conn.close()
        return json.dumps({"doc_id": did, "status": r})

    def _mem_feedback(self, args: dict) -> str:
        did = args["doc_id"]
        if not did.startswith("memory:"):
            return json.dumps({"error": "mem_feedback only removes memory facts"})
        conn = self._open()
        try:
            n = ui.delete_doc(conn, did)
        finally:
            conn.close()
        return json.dumps({"removed_chunks": n, "doc_id": did})

    # -- write hooks (gated on primary context) -----------------------------
    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if ui is None or not self._writable:
            return
        try:
            if action == "add" and content:
                did = _fact_doc_id(content)
                conn = self._open()
                try:
                    ui.upsert_doc(conn, doc_id=did, source="memory",
                                  text=content, mtime=time.time(), kind="fact")
                finally:
                    conn.close()
            elif action == "remove" and content:
                conn = self._open()
                try:
                    ui.delete_doc(conn, _fact_doc_id(content))
                finally:
                    conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("mnemosyne on_memory_write mirror failed: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if ui is None or not self._writable:
            return
        if not self._config.get("auto_extract", False) or not messages:
            return
        self._auto_extract(messages)

    _PREF = [
        re.compile(r"\bI\s+(?:prefer|like|love|use|want|need|always|never|usually)\s+(.+)", re.I),
        re.compile(r"\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)", re.I),
    ]
    _DEC = [
        re.compile(r"\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)", re.I),
        re.compile(r"\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)", re.I),
    ]

    def _auto_extract(self, messages: list) -> None:
        n = 0
        try:
            conn = self._open()
            try:
                for msg in messages:
                    if msg.get("role") != "user":
                        continue
                    c = msg.get("content", "")
                    if not isinstance(c, str) or len(c) < 10:
                        continue
                    if any(p.search(c) for p in self._PREF) or any(p.search(c) for p in self._DEC):
                        fact = c[:400].strip()
                        did = _fact_doc_id(fact)
                        if ui.upsert_doc(conn, doc_id=did, source="memory",
                                         text=fact, mtime=time.time(),
                                         kind="fact") == "indexed":
                            n += 1
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("mnemosyne auto-extract failed: %s", e)
        if n:
            logger.info("mnemosyne auto-extracted %d facts", n)

    def shutdown(self) -> None:
        pass

    # -- config (local-only: no secrets, nothing to write) ------------------
    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [{
            "key": "auto_extract",
            "description": "Pattern-distil salient facts at session end",
            "default": "true", "choices": ["true", "false"],
        }]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            import yaml
            p = Path(hermes_home) / "config.yaml"
            existing = {}
            if p.exists():
                with open(p, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["mnemosyne"] = values
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass


def register(ctx) -> None:
    ctx.register_memory_provider(MnemosyneMemoryProvider(config=_load_plugin_config()))
