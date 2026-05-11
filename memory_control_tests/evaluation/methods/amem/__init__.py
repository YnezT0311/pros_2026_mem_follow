import os
from typing import Any, Dict, List

from ..base import MethodAdapter
from ..utils import load_official_amem_module
from ...shared import build_eval_prompt, ensure_openai_env, load_openai_client, request_text


class AMemAdapter(MethodAdapter):
    backend_name = "a_mem_retrieval"

    def __init__(
        self,
        *,
        amem_impl: Any,
        client: Any,
        model: str,
        memory_limit: int,
        persona_messages: List[Dict[str, str]],
    ) -> None:
        self.amem_impl = amem_impl
        self.client = client
        self.model = model
        self.memory_limit = memory_limit
        self.persona_messages = persona_messages
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
        memory_count = len(getattr(self.amem_impl.memory_system, "memories", {}))
        self.preload_log["memory_count"] = memory_count
        step_log["memory_count_after_stage"] = memory_count
        self.preload_log["preload_steps"].append(step_log)

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
        response = request_text(self.client, self.model, messages)
        return {
            "model_response": response,
            "retrieved_memories": {
                "query_keywords": keyword_text,
                "raw_context": retrieved,
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "amem_source": "vendored_official_amem",
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
    return AMemAdapter(
        amem_impl=amem_impl,
        client=client,
        model=args.model,
        memory_limit=args.memory_limit,
        persona_messages=persona_messages,
    )
