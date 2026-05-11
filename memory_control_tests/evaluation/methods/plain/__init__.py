"""Plain (no external memory) adapter.

Sends the truncated baseline conversation directly to the model and asks the
MCQ. This is the reference condition for all memory-system comparisons; it
also serves as the "no-memory" arm in the instruction-control summary.
"""

from typing import Any, Dict, List

from ..base import MethodAdapter
from ...shared import build_eval_prompt, load_openai_client, mark_cache_breakpoint, request_text


class PlainAdapter(MethodAdapter):
    backend_name = "plain"
    supports_parallel_mcq = True

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        persona_messages: List[Dict[str, str]],
        reasoning_effort: str = "",
    ) -> None:
        self.client = client
        self.model = model
        self.persona_messages = persona_messages
        self.reasoning_effort = reasoning_effort or None
        self.context_messages: List[Dict[str, str]] = []

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        del stage_batches, ask_period
        self.context_messages = list(context_messages)

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        prompt = build_eval_prompt(question, choices)
        cached_prefix = mark_cache_breakpoint(self.persona_messages + self.context_messages)
        messages = cached_prefix + [{"role": "user", "content": prompt}]
        response = request_text(
            self.client, self.model, messages,
            reasoning_effort=self.reasoning_effort,
        )
        return {"model_response": response, "retrieved_memories": None}

    def debug_payload(self) -> Dict[str, Any]:
        return {"context_message_count": len(self.context_messages)}


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> PlainAdapter:
    client = load_openai_client(args.api_key_file)
    return PlainAdapter(
        client=client,
        model=args.model,
        persona_messages=persona_messages,
        reasoning_effort=getattr(args, "reasoning_effort", "") or "",
    )
