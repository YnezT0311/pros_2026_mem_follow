import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import MethodAdapter
from ...shared import build_eval_prompt, ensure_openai_env, load_openai_client, request_text
from ._client import load_local_mem0_memory, reset_runtime_root
from ._patches import (
    format_memories,
    run_add_with_llm_trace,
    snapshot_store,
)


class Mem0Adapter(MethodAdapter):
    backend_name = "mem0_retrieval"

    def __init__(
        self,
        *,
        memory: Any,
        client: Any,
        model: str,
        user_id: str,
        run_id: str,
        memory_limit: int,
        persona_messages: List[Dict[str, str]],
    ) -> None:
        self.memory = memory
        self.client = client
        self.model = model
        self.user_id = user_id
        self.run_id = run_id
        self.memory_limit = memory_limit
        self.persona_messages = persona_messages
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
        # stage. (mem0 batches one extractor call per add(), so we keep the
        # batched call to preserve behavior; we don't split per-turn.)
        try:
            pre_snapshot = snapshot_store(
                self.memory, user_id=self.user_id, run_id=self.run_id, limit=200
            )
        except Exception as exc:
            pre_snapshot = None
            pre_snapshot_error = repr(exc)
        else:
            pre_snapshot_error = None

        add_result, llm_call_trace = run_add_with_llm_trace(
            self.memory,
            clean_messages,
            user_id=self.user_id,
            run_id=self.run_id,
        )
        self.preload_log["add_result"] = add_result
        self.preload_log["llm_call_trace"].extend(llm_call_trace)
        step_log: Dict[str, Any] = {
            "stage": stage_label,
            "input_messages": clean_messages,
            "add_result": add_result,
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
        if isinstance(add_result, dict):
            for r in (add_result.get("results") or []):
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
        prompt = build_eval_prompt(question, choices)
        memories_text = format_memories(search_result)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": f"Retrieved memories:\n{memories_text}\n\n{prompt}",
            },
        ]
        response = request_text(self.client, self.model, messages)
        return {
            "model_response": response,
            "retrieved_memories": search_result,
        }

    def debug_payload(self) -> Dict[str, Any]:
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "mem0_source": "self_hosted_with_memoryctrl_patches",
        }


def _resolve_runtime_root(args: Any) -> Path:
    explicit = getattr(args, "mem0_runtime_root", "") or ""
    if explicit:
        return Path(explicit)
    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
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

    memory = load_local_mem0_memory(runtime_root)
    client = load_openai_client(args.api_key_file)
    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
    user_id = f"{args.world}_{stem}"
    run_id = stem
    return Mem0Adapter(
        memory=memory,
        client=client,
        model=args.model,
        user_id=user_id,
        run_id=run_id,
        memory_limit=args.memory_limit,
        persona_messages=persona_messages,
    )
