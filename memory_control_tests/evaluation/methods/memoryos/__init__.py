"""MemoryOS (3-tier short / mid / long) adapter.

The vendored MemoryOS adapter layers MemoryOS into the
MemoryCtrl evaluation pipeline:

  * Preload: pair each user turn with the next assistant turn into a single QA
    page and call ``Memoryos.add_memory(user_input, agent_response, timestamp)``.
    MemoryOS handles short-term FIFO eviction → mid-term topic-clustered
    sessions → long-term profile / knowledge extraction internally; we just
    feed it pages in order.
  * Answer: ``Memoryos.retriever.retrieve_context`` is read-only and returns
    ``retrieved_pages`` (mid-term), ``retrieved_user_knowledge``,
    ``retrieved_assistant_knowledge``. We then build the standard MCQ prompt
    via ``build_eval_prompt`` and route the final answer call through
    OpenRouter (so all methods share one model surface). MemoryOS's own
    ``get_response`` is bypassed because it would call ``add_memory`` again
    and pollute downstream stages (CLAUDE.md rule 3).

NOTE (CLAUDE.md rule 4 – assistant turns):
    MemoryOS pairs (user, assistant) into ONE page via add_memory's signature
    ``(user_input, agent_response, timestamp)``. Feeding the assistant turn
    therefore does NOT increase short-term FIFO pressure or mid-term promotion
    frequency — the cost is identical to user-only — so we feed both.

NOTE (internal LLM):
    Every internal MemoryOS LLM call (CONTINUITY_CHECK, META_INFO,
    MULTI_SUMMARY, PERSONALITY_ANALYSIS, KNOWLEDGE_EXTRACTION) goes through
    its private ``OpenAIClient`` — we point that at OpenRouter so MemoryOS's
    write-time LLM usage shows up on the same key as the answer model.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import MethodAdapter
from ..utils import load_official_memoryos_module
from ...shared import (
    build_eval_prompt,
    ensure_openai_env,
    load_openai_client,
    load_openai_credentials,
    request_text,
    resolve_model_name,
)


def _safe_token(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return cleaned.strip("_") or "default"


def _new_token_log() -> Dict[str, Dict[str, int]]:
    return {
        bucket: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "failed": 0}
        for bucket in ("internal", "answer")
    }


def _install_token_counter(openai_client: Any, usage_log: Dict[str, int]) -> Any:
    """Wrap `openai_client.chat.completions.create` so each call's
    `response.usage` accumulates into ``usage_log``. MemoryOS's
    ``OpenAIClient.chat_completion`` is a thin wrapper around this method,
    so patching here catches every internal LLM call (CONTINUITY_CHECK,
    META_INFO, MULTI_SUMMARY, PERSONALITY_ANALYSIS, KNOWLEDGE_EXTRACTION,
    UPDATE_PROFILE, etc.)."""
    original_create = openai_client.chat.completions.create

    def counting_create(*args: Any, **kwargs: Any):
        try:
            resp = original_create(*args, **kwargs)
        except Exception:
            usage_log["calls"] += 1
            usage_log["failed"] += 1
            raise
        usage_log["calls"] += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            usage_log["prompt_tokens"] += pt
            usage_log["completion_tokens"] += ct
            usage_log["total_tokens"] += pt + ct
        return resp

    openai_client.chat.completions.create = counting_create
    return original_create


def _build_counting_answer_call(client: Any, resolved_model: str, usage_log: Dict[str, int]) -> Any:
    """OpenRouter answer-call wrapper that records token usage. Separates the
    final MCQ-answer cost from MemoryOS's internal write-time LLM cost."""

    def _call(messages: List[Dict[str, str]]) -> str:
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            usage_log["calls"] += 1
            usage_log["failed"] += 1
            print(f"[memoryos] OpenRouter answer call failed: {exc}", flush=True)
            return ""
        usage_log["calls"] += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            usage_log["prompt_tokens"] += pt
            usage_log["completion_tokens"] += ct
            usage_log["total_tokens"] += pt + ct
        try:
            return (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, TypeError):
            return ""

    return _call


def _pair_turns(messages: List[Dict[str, str]], start_ts: str = "2026-04-01 00:00") -> List[Dict[str, str]]:
    """Pair consecutive (user → assistant) turns into a single QA page.

    Two consecutive user turns produce a page with empty agent_response; an
    orphan assistant turn (no preceding user) produces a page with empty
    user_input. Mirrors the canonical LOCOMO ``process_conversation`` shape so
    MemoryOS's continuity-detection and meta-info generation see the same
    structure they were designed for.
    """
    pages: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            pages.append({"user_input": content, "agent_response": "", "timestamp": start_ts})
        elif role == "assistant":
            if pages and not pages[-1]["agent_response"]:
                pages[-1]["agent_response"] = content
            else:
                pages.append({"user_input": "", "agent_response": content, "timestamp": start_ts})
        # Other roles (system, etc.) are ignored — they're never in context_messages.
    return pages


class MemoryOSAdapter(MethodAdapter):
    backend_name = "memoryos_retrieval"

    def __init__(
        self,
        *,
        memoryos_module: Any,
        client: Any,
        model: str,
        resolved_model: str,
        user_id: str,
        data_storage_path: str,
        short_term_capacity: int,
        mid_term_capacity: int,
        long_term_knowledge_capacity: int,
        retrieval_queue_capacity: int,
        embedding_model_name: str,
        persona_messages: List[Dict[str, str]],
        openrouter_api_key: str,
        openrouter_base_url: str,
    ) -> None:
        self.client = client
        self.model = model
        self.persona_messages = persona_messages
        self.user_id = user_id
        self.short_term_capacity = short_term_capacity

        # Build the Memoryos instance — internal LLM calls flow through this
        # instance's OpenAIClient (constructed inside Memoryos.__init__ from
        # the api_key + base_url we pass here).
        self.memoryos = memoryos_module.Memoryos(
            user_id=user_id,
            openai_api_key=openrouter_api_key,
            data_storage_path=data_storage_path,
            openai_base_url=openrouter_base_url,
            short_term_capacity=short_term_capacity,
            mid_term_capacity=mid_term_capacity,
            long_term_knowledge_capacity=long_term_knowledge_capacity,
            retrieval_queue_capacity=retrieval_queue_capacity,
            llm_model=resolved_model,
            embedding_model_name=embedding_model_name,
        )
        # Token counter: wrap the underlying OpenAI client inside Memoryos's
        # OpenAIClient wrapper so every internal call (continuity check,
        # meta info, multi-summary, personality + knowledge extraction,
        # update profile) accumulates usage. Final-answer call is a separate
        # bucket via _build_counting_answer_call below.
        self.token_log = _new_token_log()
        try:
            _install_token_counter(self.memoryos.client.client, self.token_log["internal"])
        except AttributeError:
            print("[memoryos] WARNING: could not install token counter — memoryos.client.client missing", flush=True)
        self.answer_call = _build_counting_answer_call(client, resolved_model, self.token_log["answer"])
        self.preload_log: Dict[str, Any] = {
            "input_messages": [],
            "preload_steps": [],
            "written_pages": [],  # per-page (user_input, agent_response, stage, timestamp) we asked MemoryOS to write
            "session_scope": {"user_id": user_id, "data_storage_path": data_storage_path},
            "config": {
                "short_term_capacity": short_term_capacity,
                "mid_term_capacity": mid_term_capacity,
                "long_term_knowledge_capacity": long_term_knowledge_capacity,
                "retrieval_queue_capacity": retrieval_queue_capacity,
                "embedding_model_name": embedding_model_name,
                "resolved_model": resolved_model,
            },
        }

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        del ask_period
        if stage_batches:
            for batch in stage_batches:
                self._preload_one_stage(batch["messages"], stage_label=batch.get("period", ""))
        else:
            self._preload_one_stage(context_messages, stage_label="")

    def _preload_one_stage(self, messages: List[Dict[str, str]], stage_label: str = "") -> None:
        pages = _pair_turns(messages)
        if not pages:
            return
        self.preload_log["input_messages"].extend(pages)
        step_log: Dict[str, Any] = {
            "stage": stage_label,
            "page_count": len(pages),
            "first_page": pages[0],
            "last_page": pages[-1],
        }
        for idx, page in enumerate(pages):
            self.memoryos.add_memory(
                user_input=page["user_input"],
                agent_response=page["agent_response"],
                timestamp=page["timestamp"],
            )
            # Record what we asked MemoryOS to write so the error-analysis
            # report can show per-turn stored content. This logs the *input*
            # pair (user/agent) — MemoryOS may further compress/extract from
            # it, but the pair is what we know was committed for storage.
            self.preload_log["written_pages"].append({
                "stage": stage_label,
                "user_input": page["user_input"],
                "agent_response": page["agent_response"],
                "timestamp": page["timestamp"],
                "content": (
                    f"User: {page['user_input']}\n"
                    f"Assistant: {page['agent_response']}"
                ),
            })
            if idx == 0 or idx == len(pages) - 1 or (idx + 1) % 10 == 0:
                print(
                    f"[memoryos] preload {stage_label or 'stage'}: {idx + 1}/{len(pages)} pages",
                    flush=True,
                )
        try:
            stats = self.memoryos.get_memory_stats()
            step_log["post_stage_stats"] = stats
        except Exception as exc:
            step_log["post_stage_stats_error"] = repr(exc)
        # Dump everything MemoryOS extracted from the conversation so far:
        # short-term FIFO, mid-term topic-summarized sessions, and the long-term
        # knowledge entries + user profile that the heat-triggered LLM analyses
        # produced. Stored once per stage; the final stage is a strict superset.
        step_log["store_snapshot"] = self._snapshot_memoryos_state()
        self.preload_log["preload_steps"].append(step_log)
        self.preload_log["store_snapshot"] = step_log["store_snapshot"]

    def _snapshot_memoryos_state(self) -> Dict[str, Any]:
        """Read-only dump of every layer's contents so the report can show
        exactly what MemoryOS distilled from the conversation."""
        out: Dict[str, Any] = {}
        try:
            st = self.memoryos.short_term_memory
            out["short_term"] = list(st.get_all())
        except Exception as exc:
            out["short_term_error"] = repr(exc)
        try:
            mt = self.memoryos.mid_term_memory
            sessions_dump = []
            for sid, sess in (mt.sessions or {}).items():
                sessions_dump.append({
                    "sid": sid,
                    "summary": sess.get("summary", ""),
                    "keywords": list(sess.get("keywords", []) or []),
                    "H_segment": sess.get("H_segment"),
                    "N_visit": sess.get("N_visit"),
                    "page_count": len(sess.get("details", []) or []),
                })
            out["mid_term_sessions"] = sessions_dump
        except Exception as exc:
            out["mid_term_error"] = repr(exc)
        try:
            ltm_user = self.memoryos.user_long_term_memory
            out["user_profile"] = ltm_user.get_raw_user_profile(self.user_id) or ""
            knowledge_list = []
            for entry in (ltm_user.knowledge_base or []):
                if isinstance(entry, dict):
                    knowledge_list.append({
                        "knowledge": entry.get("knowledge", ""),
                        "timestamp": entry.get("timestamp", ""),
                    })
            out["user_knowledge"] = knowledge_list
        except Exception as exc:
            out["user_long_term_error"] = repr(exc)
        try:
            ltm_asst = self.memoryos.assistant_long_term_memory
            asst_list = []
            for entry in (ltm_asst.knowledge_base or []):
                if isinstance(entry, dict):
                    asst_list.append({
                        "knowledge": entry.get("knowledge", ""),
                        "timestamp": entry.get("timestamp", ""),
                    })
            out["assistant_knowledge"] = asst_list
        except Exception as exc:
            out["assistant_long_term_error"] = repr(exc)
        return out

    def _format_retrieved(self, retrieval: Dict[str, Any]) -> str:
        retrieved_pages = retrieval.get("retrieved_pages", []) or []
        retrieved_user_knowledge = retrieval.get("retrieved_user_knowledge", []) or []
        retrieved_assistant_knowledge = retrieval.get("retrieved_assistant_knowledge", []) or []

        try:
            user_profile_text = self.memoryos.user_long_term_memory.get_raw_user_profile(self.user_id)
        except Exception:
            user_profile_text = ""
        if not user_profile_text or str(user_profile_text).strip().lower() == "none":
            user_profile_text = "No detailed profile available yet."

        sections: List[str] = []
        sections.append(f"【User Profile】\n{user_profile_text}")

        if retrieved_assistant_knowledge:
            ak_lines = [
                f"  - {ak.get('knowledge', '')} (Recorded: {ak.get('timestamp', '')})"
                for ak in retrieved_assistant_knowledge
            ]
            sections.append("【Assistant Knowledge】\n" + "\n".join(ak_lines))

        if retrieved_pages:
            page_lines = []
            for page in retrieved_pages:
                page_lines.append(
                    f"【Historical Memory】\n"
                    f"  User: {page.get('user_input', '')}\n"
                    f"  Assistant: {page.get('agent_response', '')}\n"
                    f"  Time: {page.get('timestamp', '')}\n"
                    f"  Conversation chain overview: {page.get('meta_info', 'N/A')}"
                )
            sections.append("\n\n".join(page_lines))

        if retrieved_user_knowledge:
            uk_lines = [
                f"  - {uk.get('knowledge', '')} (Recorded: {uk.get('timestamp', '')})"
                for uk in retrieved_user_knowledge
            ]
            sections.append("【Relevant User Knowledge】\n" + "\n".join(uk_lines))

        return "\n\n".join(sections) if sections else "No relevant memories were retrieved."

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        retrieval = self.memoryos.retriever.retrieve_context(
            user_query=question,
            user_id=self.user_id,
        )
        memories_text = self._format_retrieved(retrieval)
        prompt = build_eval_prompt(question, choices)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": f"Retrieved memories:\n{memories_text}\n\n{prompt}",
            }
        ]
        response = self.answer_call(messages)
        return {
            "model_response": response,
            "retrieved_memories": {
                "retrieved_pages": retrieval.get("retrieved_pages", []),
                "retrieved_user_knowledge": retrieval.get("retrieved_user_knowledge", []),
                "retrieved_assistant_knowledge": retrieval.get("retrieved_assistant_knowledge", []),
                "formatted_text": memories_text,
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        internal = self.token_log["internal"]
        answer = self.token_log["answer"]
        return {
            "preload": self.preload_log,
            "memoryos_source": "vendored_memoryos_pypi",
            "token_usage": {
                "model": getattr(self.memoryos, "llm_model", ""),
                "internal": dict(internal),
                "answer": dict(answer),
                "total": {
                    "calls": internal.get("calls", 0) + answer.get("calls", 0),
                    "prompt_tokens": internal.get("prompt_tokens", 0) + answer.get("prompt_tokens", 0),
                    "completion_tokens": internal.get("completion_tokens", 0) + answer.get("completion_tokens", 0),
                    "total_tokens": internal.get("total_tokens", 0) + answer.get("total_tokens", 0),
                    "failed": internal.get("failed", 0) + answer.get("failed", 0),
                },
            },
        }


def _resolve_runtime_root(args: Any) -> Path:
    explicit = getattr(args, "memoryos_runtime_root", "") or ""
    if explicit:
        return Path(explicit)
    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
    world = getattr(args, "world", "baseline")
    return Path("data/runtime/memoryos") / world / stem


def _reset_runtime_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> MemoryOSAdapter:
    ensure_openai_env(args.api_key_file)
    api_key, base_url = load_openai_credentials(args.api_key_file)
    os.environ["MODEL"] = args.model
    embedding_model_name = getattr(args, "embedding_model", "") or "all-MiniLM-L6-v2"
    if embedding_model_name:
        os.environ["EMBEDDING_MODEL"] = embedding_model_name

    runtime_root = _resolve_runtime_root(args)
    if getattr(args, "memoryos_reset_runtime", True):
        _reset_runtime_root(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    memoryos_module = load_official_memoryos_module()
    client = load_openai_client(args.api_key_file)

    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
    user_id = _safe_token(f"{getattr(args, 'world', 'baseline')}_{stem}")
    resolved_model = resolve_model_name(args.model)

    return MemoryOSAdapter(
        memoryos_module=memoryos_module,
        client=client,
        model=args.model,
        resolved_model=resolved_model,
        user_id=user_id,
        data_storage_path=str(runtime_root),
        short_term_capacity=getattr(args, "memoryos_short_term_capacity", 10),
        mid_term_capacity=getattr(args, "memoryos_mid_term_capacity", 2000),
        long_term_knowledge_capacity=getattr(args, "memoryos_long_term_knowledge_capacity", 100),
        retrieval_queue_capacity=getattr(args, "memoryos_retrieval_queue_capacity", 7),
        embedding_model_name=embedding_model_name,
        persona_messages=persona_messages,
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
    )
