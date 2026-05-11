"""Zep adapter — mirrors mem0_official/evaluation/src/zep/{add,search}.py
semantically, translated to the current zep_cloud SDK.

Translation summary:

  * `client.memory.add_session(user_id, session_id)` is gone in zep_cloud 3.x;
    use `client.thread.create(thread_id=..., user_id=...)` instead.
  * `client.memory.add(session_id, messages=[Message(...)])` is gone; use
    `client.thread.add_messages(thread_id=..., messages=[Message(...)])`.
  * Retrieval (`graph.search(scope='edges'|'nodes', reranker=...)`) is
    unchanged from the official implementation.

Content format follows the official:
    Message(role=<speaker>, role_type='user', content=f'{timestamp}: {text}')

The adapter is also safe to call `preload(...)` multiple times so the same
underlying graph is reused across `ask_period` checkpoints in a progressive
sweep — see the principle in CLAUDE.md "Memory is incremental across stages."
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import MethodAdapter
from ...shared import build_eval_prompt, ensure_openai_env, load_openai_client, request_text


_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}

"""


_SIDE_NOTE_RE = re.compile(r"^Side_Note:\s*\[(.*?)\]\s*(.+?)\s*$")


def _parse_side_note(line: str) -> Optional[Tuple[str, str]]:
    """Returns (event_text, timestamp) if `line` is a Side_Note, else None."""
    m = _SIDE_NOTE_RE.match((line or "").strip())
    return (m.group(1), m.group(2)) if m else None


def _format_edge_date_range(edge: Any) -> str:
    valid_at = getattr(edge, "valid_at", None) or "date unknown"
    invalid_at = getattr(edge, "invalid_at", None) or "present"
    return f"{valid_at} - {invalid_at}"


def _compose_search_context(edges: List[Any], nodes: List[Any]) -> str:
    """Same shape as mem0_official `ZepSearch.compose_search_context`."""
    facts = [f"  - {getattr(e, 'fact', '')} ({_format_edge_date_range(e)})" for e in edges]
    entities = [f"  - {getattr(n, 'name', '')}: {getattr(n, 'summary', '')}" for n in nodes]
    return _TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))


_PERIODS = (
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
)


def _walk_timestamped_messages(transformed: Dict[str, Any], ask_period: str) -> List[Tuple[str, str, str]]:
    """Yield (timestamp, role, content) up to and including `ask_period`.

    Side_Notes mark timestamp boundaries; subsequent User:/Assistant: turns
    inherit the most recently seen timestamp. Turns before any Side_Note in a
    stage carry an empty timestamp (we still emit them).
    """
    if ask_period not in _PERIODS:
        return []
    end = _PERIODS.index(ask_period)
    out: List[Tuple[str, str, str]] = []
    current_ts = ""
    for period in _PERIODS[: end + 1]:
        for line in transformed.get(period, []):
            if not isinstance(line, str):
                continue
            sn = _parse_side_note(line)
            if sn:
                current_ts = sn[1]
                continue
            if line.startswith("User:"):
                out.append((current_ts, "user", line[len("User:"):].strip()))
            elif line.startswith("Assistant:"):
                out.append((current_ts, "assistant", line[len("Assistant:"):].strip()))
            elif line.strip():
                out.append((current_ts, "user", line.strip()))
    return out


class ZepAdapter(MethodAdapter):
    backend_name = "zep_retrieval"

    def __init__(
        self,
        *,
        client: Any,
        zep_client: Any,
        model: str,
        user_id: str,
        session_id: str,
        transformed_conversation: Dict[str, Any],
    ) -> None:
        from zep_cloud import Message  # pylint: disable=import-outside-toplevel

        self._ZepMessage = Message
        self.client = client
        self.zep_client = zep_client
        self.model = model
        self.user_id = user_id
        self.session_id = session_id
        self.transformed_conversation = transformed_conversation
        self._written_count = 0
        self.preload_log: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "input_messages": [],
            "loaded_messages": [],
        }

        # Idempotent setup: user + thread exist before any preload call.
        try:
            zep_client.user.add(user_id=user_id)
        except Exception:
            pass  # user already exists
        try:
            zep_client.thread.create(thread_id=session_id, user_id=user_id)
        except Exception:
            pass  # thread already exists

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        del stage_batches, context_messages
        # Filter to user turns only — Zep bills per byte ingested.
        all_timestamped = [
            (ts, role, content)
            for ts, role, content in _walk_timestamped_messages(self.transformed_conversation, ask_period)
            if role == "user"
        ]
        new_messages = all_timestamped[self._written_count:]
        self._written_count = len(all_timestamped)

        for ts, role, content in new_messages:
            if not content:
                continue
            serialized = f"{ts}: {content}" if ts else content
            self.zep_client.thread.add_messages(
                thread_id=self.session_id,
                messages=[self._ZepMessage(role=role, role_type="user", content=serialized)],
            )
            self.preload_log["input_messages"].append({"timestamp": ts, "role": role, "content": content})
            self.preload_log["loaded_messages"].append(serialized)

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        edges_results = self.zep_client.graph.search(
            user_id=self.user_id, reranker="cross_encoder", query=question, scope="edges", limit=20,
        ).edges
        node_results = self.zep_client.graph.search(
            user_id=self.user_id, reranker="rrf", query=question, scope="nodes", limit=20,
        ).nodes
        context = _compose_search_context(edges_results, node_results)
        prompt = build_eval_prompt(question, choices)
        response = request_text(
            self.client,
            self.model,
            [{"role": "user", "content": f"Retrieved memories:\n{context}\n\n{prompt}"}],
        )
        return {
            "model_response": response,
            "retrieved_memories": {
                "context": context,
                "edges": [
                    {
                        "fact": getattr(e, "fact", ""),
                        "valid_at": getattr(e, "valid_at", None),
                        "invalid_at": getattr(e, "invalid_at", None),
                    }
                    for e in edges_results
                ],
                "nodes": [
                    {"name": getattr(n, "name", ""), "summary": getattr(n, "summary", "")}
                    for n in node_results
                ],
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        return {
            "preload": self.preload_log,
            "zep_source": "memoryctrl_native_new_sdk",
            "written_count": self._written_count,
        }

    def close(self) -> None:
        try:
            self.zep_client.thread.delete(thread_id=self.session_id)
        except Exception:
            pass


def build_adapter(
    *,
    args: Any,
    transformed_conversation: Dict[str, Any],
    **_: Any,
) -> ZepAdapter:
    ensure_openai_env(args.api_key_file)
    os.environ["MODEL"] = args.model

    zep_key_file = getattr(args, "zep_api_key_file", "keys/zep_api_key.txt")
    if Path(zep_key_file).exists() and not os.getenv("ZEP_API_KEY", "").strip():
        os.environ["ZEP_API_KEY"] = Path(zep_key_file).read_text(encoding="utf-8").strip()
    if not os.getenv("ZEP_API_KEY", "").strip():
        raise RuntimeError(
            "ZEP_API_KEY not set. Provide it via env or keys/zep_api_key.txt."
        )

    from zep_cloud.client import Zep  # pylint: disable=import-outside-toplevel

    zep_client = Zep(api_key=os.environ["ZEP_API_KEY"])
    client = load_openai_client(args.api_key_file)

    stem = Path(args.rendered).stem
    model_tag = "".join(ch if ch.isalnum() else "_" for ch in str(args.model))
    world_tag = "".join(ch if ch.isalnum() else "_" for ch in str(args.world))
    user_id = f"memoryctrl_user_{world_tag}_{model_tag}_{stem}"
    session_id = f"memoryctrl_session_{world_tag}_{model_tag}_{stem}"

    return ZepAdapter(
        client=client,
        zep_client=zep_client,
        model=args.model,
        user_id=user_id,
        session_id=session_id,
        transformed_conversation=transformed_conversation,
    )
