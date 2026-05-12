import os
from typing import Any, Dict, List

from ..base import MethodAdapter
from ..utils import load_official_amem_module
from ...shared import build_eval_prompt, ensure_openai_env, load_openai_client, request_text, resolve_model_name


def _new_token_log() -> Dict[str, Dict[str, int]]:
    return {
        bucket: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "failed": 0}
        for bucket in ("internal", "answer")
    }


def _build_counting_answer_call(client: Any, resolved_model: str, usage_log: Dict[str, int]) -> Any:
    """Replacement for `request_text` that records token usage on the final
    answer call so the adapter can report cost per phase (internal vs answer).
    """

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
            print(f"[a-mem] OpenRouter answer call failed: {exc}", flush=True)
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


def _install_token_counter(client: Any, usage_log: Dict[str, int]) -> Any:
    """Wrap `client.chat.completions.create` so each call's response.usage is
    accumulated into ``usage_log``. Returns the original create function so
    the caller can restore it on teardown (we don't strictly need to here —
    the adapter and client are 1:1 per evaluation run)."""
    original_create = client.chat.completions.create

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

    client.chat.completions.create = counting_create
    return original_create


def _snapshot_amem_notes(memories: Any) -> List[Dict[str, Any]]:
    """Dump every RobustMemoryNote's LLM-analyzed metadata so post-hoc
    inspection can see what the evolution / analyze_content steps actually
    wrote (content stays immutable, but context / keywords / tags / links
    drift across UPDATE_NEIGHBOR calls)."""
    out: List[Dict[str, Any]] = []
    if not isinstance(memories, dict):
        return out
    for note_id, note in memories.items():
        out.append(
            {
                "id": getattr(note, "id", note_id),
                "content": getattr(note, "content", ""),
                "context": getattr(note, "context", ""),
                "keywords": list(getattr(note, "keywords", []) or []),
                "tags": list(getattr(note, "tags", []) or []),
                "links": list(getattr(note, "links", []) or []),
                "timestamp": getattr(note, "timestamp", ""),
            }
        )
    return out


class AMemAdapter(MethodAdapter):
    backend_name = "a_mem_retrieval"

    def __init__(
        self,
        *,
        amem_impl: Any,
        client: Any,
        model: str,
        resolved_model: str,
        memory_limit: int,
        persona_messages: List[Dict[str, str]],
    ) -> None:
        self.amem_impl = amem_impl
        self.client = client
        self.model = model
        self.resolved_model = resolved_model
        self.memory_limit = memory_limit
        self.persona_messages = persona_messages
        # Token counter for every internal A-Mem LLM call (analyze_content,
        # evolution_decision, strengthen, update_neighbors, query_keywords).
        # Wired via _install_token_counter in build_adapter (below).
        self.token_log = _new_token_log()
        # Wrap the final-answer request_text call too, so we get a per-bucket
        # breakdown (internal vs answer) like mem0 / MemTree adapters do.
        self.answer_call = _build_counting_answer_call(client, resolved_model, self.token_log["answer"])
        self.preload_log: Dict[str, Any] = {
            "input_messages": [],
            "written_notes": [],
            "memory_count": 0,
            "preload_steps": [],
        }

    def _reset_log(self) -> None:
        self.preload_log = {
            "input_messages": [],
            "written_notes": [],
            "memory_count": 0,
            "preload_steps": [],
        }

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        self.amem_impl.reset()
        self._reset_log()
        if stage_batches:
            for batch in stage_batches:
                self._preload_one_stage(batch["messages"], stage_label=batch["period"])
        else:
            self._preload_one_stage(context_messages, stage_label=ask_period)

    def _preload_one_stage(self, messages: List[Dict[str, str]], stage_label: str = "") -> None:
        step_log: Dict[str, Any] = {
            "stage_label": stage_label,
            "input_messages": [],
            "written_notes": [],
            "memory_count_after_stage": 0,
        }
        total_messages = sum(1 for msg in messages if msg.get("content", "").strip())
        written_in_stage = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            note_content = f"Speaker {role} says : {content}"
            note_time = f"2025010100{len(self.preload_log['written_notes']) + 1:02d}"
            payload = {"role": role, "content": content, "time": note_time}
            self.preload_log["input_messages"].append(payload)
            step_log["input_messages"].append(payload)
            self.amem_impl.add_memory(content=note_content, time=note_time)
            note_log = {"content": note_content, "time": note_time}
            self.preload_log["written_notes"].append(note_log)
            step_log["written_notes"].append(note_log)
            written_in_stage += 1
            if written_in_stage == 1 or written_in_stage == total_messages or written_in_stage % 10 == 0:
                print(
                    f"[a-mem] preload {stage_label or 'stage'}: {written_in_stage}/{total_messages} notes",
                    flush=True,
                )
        memories_dict = getattr(self.amem_impl.memory_system, "memories", {})
        memory_count = len(memories_dict)
        self.preload_log["memory_count"] = memory_count
        step_log["memory_count_after_stage"] = memory_count
        # Dump the LLM-analyzed metadata (keywords/context/tags/links) for every
        # note currently in the store. This is what `analyze_content` and the
        # evolution step actually wrote — without this dump we couldn't tell
        # whether STRENGTHEN/UPDATE_NEIGHBOR fired, or what tags got assigned.
        step_log["store_snapshot"] = _snapshot_amem_notes(memories_dict)
        self.preload_log["preload_steps"].append(step_log)
        # Keep only the latest stage's snapshot at the top level (each new
        # stage is a strict superset; saving every one would inflate the file).
        self.preload_log["store_snapshot"] = step_log["store_snapshot"]

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        keyword_text = self.amem_impl.generate_query_keywords(question)
        retrieved = self.amem_impl.search_memory(keyword_text, k=self.memory_limit)
        memories_text = retrieved if retrieved.strip() else "No relevant memories were retrieved."
        prompt = build_eval_prompt(question, choices)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": f"Retrieved memories:\n{memories_text}\n\n{prompt}",
            },
        ]
        response = self.answer_call(messages)
        return {
            "model_response": response,
            "retrieved_memories": {
                "query_keywords": keyword_text,
                "raw_context": retrieved,
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        internal = self.token_log["internal"]
        answer = self.token_log["answer"]
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "amem_source": "vendored_official_amem",
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


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> AMemAdapter:
    ensure_openai_env(args.api_key_file)
    os.environ["MODEL"] = args.model
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    amem_module = load_official_amem_module()
    embedding_model = args.embedding_model or "all-MiniLM-L6-v2"
    amem_impl = amem_module.AMem(
        model=args.model,
        embedding_model=embedding_model,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    client = load_openai_client(args.api_key_file)
    resolved_model = resolve_model_name(args.model)
    adapter = AMemAdapter(
        amem_impl=amem_impl,
        client=client,
        model=args.model,
        resolved_model=resolved_model,
        memory_limit=args.memory_limit,
        persona_messages=persona_messages,
    )
    # Wire the internal-LLM counter onto BOTH controllers A-Mem uses:
    # - memory_system.llm_controller (drives analyze_content, evolution, etc.)
    # - _retriever_llm (drives generate_query_keywords at MCQ time)
    # Both wrap the same OpenAI client class internally; patching their
    # .client.chat.completions.create catches every internal call.
    for controller_path in (
        getattr(amem_impl, "memory_system", None) and getattr(amem_impl.memory_system, "llm_controller", None),
        getattr(amem_impl, "_retriever_llm", None) and getattr(amem_impl._retriever_llm, "llm", None),
    ):
        if controller_path is not None and hasattr(controller_path, "client"):
            _install_token_counter(controller_path.client, adapter.token_log["internal"])
    return adapter
