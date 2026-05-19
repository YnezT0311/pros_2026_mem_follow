import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import MethodAdapter
from ...paths import rendered_stem
from ...shared import build_memory_eval_prompt, ensure_openai_env, load_openai_client, resolve_model_name
from ._client import load_local_mem0_memory, reset_runtime_root
from ._patches import (
    format_memories,
    run_add_with_llm_trace,
    snapshot_store,
)


def _new_token_log() -> Dict[str, Dict[str, int]]:
    return {
        bucket: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "failed": 0}
        for bucket in ("internal", "answer")
    }


def _build_counting_request_text(
    client: Any,
    model: str,
    usage_log: Dict[str, int],
) -> Any:
    """Replacement for `request_text` that captures `response.usage` so the
    adapter can report final-MCQ-answer token cost separately from mem0's
    internal extraction / update LLM calls.
    """

    def _call(messages: List[Dict[str, str]]) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            usage_log["calls"] += 1
            usage_log["failed"] += 1
            print(f"[mem0] OpenRouter answer call failed: {exc}", flush=True)
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


class Mem0Adapter(MethodAdapter):
    backend_name = "mem0_retrieval"

    def __init__(
        self,
        *,
        memory: Any,
        client: Any,
        model: str,
        resolved_model: str,
        user_id: str,
        run_id: str,
        memory_limit: int,
        preload_batch_size: int,
        persona_messages: List[Dict[str, str]],
    ) -> None:
        self.memory = memory
        self.client = client
        self.model = model
        self.resolved_model = resolved_model
        self.user_id = user_id
        self.run_id = run_id
        self.memory_limit = memory_limit
        self.preload_batch_size = max(1, preload_batch_size)
        self.persona_messages = persona_messages
        self.token_log = _new_token_log()
        self.answer_call = _build_counting_request_text(client, resolved_model, self.token_log["answer"])
        self.preload_log: Dict[str, Any] = {
            "input_messages": [],
            "infer": True,
            "llm_call_trace": [],
            "add_result": None,
            "post_add_snapshot": None,
            "post_add_snapshot_error": None,
            "preload_steps": [],
            "session_scope": {"user_id": user_id, "run_id": run_id},
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
                self._preload_one_stage(batch["messages"], stage_label=batch["period"])
        else:
            self._preload_one_stage(context_messages, stage_label="")

    def _preload_one_stage(self, messages: List[Dict[str, str]], stage_label: str = "") -> None:
        clean_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            clean_messages.append({"role": role, "content": content})
        if not clean_messages:
            return
        self.preload_log["input_messages"].extend(clean_messages)

        # Snapshot store contents BEFORE this stage's add() so the report can
        # diff against the post snapshot to attribute writes/deletes to this
        # stage.
        try:
            pre_snapshot = snapshot_store(
                self.memory, user_id=self.user_id, run_id=self.run_id, limit=200
            )
        except Exception as exc:
            pre_snapshot = None
            pre_snapshot_error = repr(exc)
        else:
            pre_snapshot_error = None

        add_results = []
        llm_call_trace = []
        for start in range(0, len(clean_messages), self.preload_batch_size):
            batch_messages = clean_messages[start : start + self.preload_batch_size]
            batch_result, batch_trace = run_add_with_llm_trace(
                self.memory,
                batch_messages,
                user_id=self.user_id,
                run_id=self.run_id,
                usage_log=self.token_log["internal"],
            )
            add_results.append(batch_result)
            llm_call_trace.extend(batch_trace)
        add_result = add_results[-1] if add_results else None
        self.preload_log["add_result"] = add_result
        self.preload_log["llm_call_trace"].extend(llm_call_trace)
        step_log: Dict[str, Any] = {
            "stage": stage_label,
            "input_messages": clean_messages,
            "add_result": add_result,
            "add_results": add_results,
            "llm_call_trace": llm_call_trace,
            "pre_add_snapshot": pre_snapshot,
            "pre_add_snapshot_error": pre_snapshot_error,
        }
        try:
            snapshot = snapshot_store(
                self.memory, user_id=self.user_id, run_id=self.run_id, limit=200
            )
            self.preload_log["post_add_snapshot"] = snapshot
            step_log["post_add_snapshot"] = snapshot
        except Exception as exc:
            error_text = repr(exc)
            self.preload_log["post_add_snapshot_error"] = error_text
            step_log["post_add_snapshot_error"] = error_text
        # Per-stage event summary: count ADD/UPDATE/DELETE/NONE events from
        # mem0's add_result so the report can show "did the backend issue any
        # delete during this stage?" at a glance.
        events_by_kind: Dict[str, List[Dict[str, str]]] = {}
        for result in add_results:
            if not isinstance(result, dict):
                continue
            for r in (result.get("results") or []):
                if not isinstance(r, dict):
                    continue
                ev = str(r.get("event", "") or "").upper()
                events_by_kind.setdefault(ev, []).append({
                    "id": str(r.get("id", "")),
                    "memory": str(r.get("memory", "")),
                    "previous_memory": str(r.get("previous_memory", "")),
                })
        step_log["events_by_kind"] = events_by_kind
        self.preload_log["preload_steps"].append(step_log)

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        search_result = self.memory.search(
            query=question,
            user_id=self.user_id,
            run_id=self.run_id,
            limit=self.memory_limit,
        )
        memories_text = format_memories(search_result)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": build_memory_eval_prompt(question, choices, memories_text),
            },
        ]
        response = self.answer_call(messages)
        return {
            "model_response": response,
            "retrieved_memories": search_result,
        }

    def debug_payload(self) -> Dict[str, Any]:
        internal = self.token_log["internal"]
        answer = self.token_log["answer"]
        llm_config = getattr(getattr(self.memory, "llm", None), "config", None)
        internal_model = getattr(llm_config, "model", None)
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "preload_batch_size": self.preload_batch_size,
            "mem0_source": "self_hosted_with_memoryctrl_patches",
            "model_routing": {
                "requested_model": self.model,
                "resolved_model": self.resolved_model,
                "internal_llm_model": internal_model,
                "answer_model": self.resolved_model,
            },
            "token_usage": {
                "model": self.resolved_model,
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
    explicit = getattr(args, "mem0_runtime_root", "") or ""
    if explicit:
        return Path(explicit)
    stem = rendered_stem(args.rendered)
    world = getattr(args, "world", "baseline")
    return Path("data/runtime/mem0") / world / stem


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> Mem0Adapter:
    ensure_openai_env(args.api_key_file)
    os.environ["MODEL"] = args.model
    if getattr(args, "embedding_model", ""):
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    runtime_root = _resolve_runtime_root(args)
    if getattr(args, "mem0_reset_runtime", True):
        reset_runtime_root(runtime_root)

    resolved_model = resolve_model_name(args.model)
    memory = load_local_mem0_memory(runtime_root, llm_model=resolved_model)
    client = load_openai_client(args.api_key_file)
    stem = rendered_stem(args.rendered)
    user_id = f"{args.world}_{stem}"
    run_id = stem
    return Mem0Adapter(
        memory=memory,
        client=client,
        model=args.model,
        resolved_model=resolved_model,
        user_id=user_id,
        run_id=run_id,
        memory_limit=args.memory_limit,
        preload_batch_size=getattr(args, "preload_batch_size", 2),
        persona_messages=persona_messages,
    )
