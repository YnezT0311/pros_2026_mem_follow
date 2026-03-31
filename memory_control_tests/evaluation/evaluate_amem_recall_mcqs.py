import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from ..transforms import apply_forget, apply_no_store, apply_no_use, build_context_messages


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "MemoryCtrl"
MODEL_ALIASES = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
}


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_api_key(api_key_file: str = "openrouter_key.txt") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and Path(api_key_file).exists():
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        legacy_key = os.getenv("OPENAI_API_KEY", "").strip()
        legacy_path = Path("openai_key.txt")
        if not legacy_key and legacy_path.exists():
            legacy_key = legacy_path.read_text(encoding="utf-8").strip()
        api_key = legacy_key
    if not api_key:
        raise FileNotFoundError("No API key found. Set OPENROUTER_API_KEY or provide openrouter_key.txt.")
    return api_key


def _load_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "").strip() or OPENROUTER_BASE_URL


def _ensure_openrouter_env(api_key_file: str = "openrouter_key.txt") -> None:
    os.environ["OPENAI_API_KEY"] = _load_api_key(api_key_file)
    os.environ["OPENAI_BASE_URL"] = _load_base_url()


def _load_openai_client(api_key_file: str = "openrouter_key.txt") -> OpenAI:
    api_key = _load_api_key(api_key_file)
    base_url = _load_base_url()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-OpenRouter-Title": OPENROUTER_TITLE},
    )


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    output = getattr(resp, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
    return ""


def _request_text(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    model = _resolve_model_name(model)
    try:
        resp = client.responses.create(model=model, input=messages)
        text = _extract_text(resp)
        if text:
            return text
    except Exception:
        pass

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def _build_eval_prompt(question: str, choices: Dict[str, str]) -> str:
    labels = list(choices.keys())
    rendered_options = [f"({label.lower()}) {choices[label]}" for label in labels]
    instructions = (
        "Find the most appropriate model response and give your final answer "
        f"{', '.join(f'({label.lower()})' for label in labels)} after the special token <final_answer>."
    )
    return f"Question: {question}\n\n{instructions}\n\n{rendered_options}"


def _extract_choice(text: str, labels: List[str]) -> str:
    cleaned = text.strip().upper()
    for label in labels:
        pattern = rf"\b({re.escape(label.upper())})\b"
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return ""


def _build_persona_system_message(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    persona = conversation.get("Expanded Persona")
    if not isinstance(persona, str) or not persona.strip():
        return []
    return [{"role": "system", "content": f"Current user persona: {persona.strip()}"}]


def _ask_period_tag(ask_period: str) -> str:
    tag = (ask_period or "").replace("Conversation ", "").replace(" Stage", "")
    return tag.strip().lower().replace(" ", "_") or "late"


def _is_valid_mcq(choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> bool:
    return bool(choices) and bool(choice_to_answer_type)


def _rate_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {
            "num_questions": 0,
            "remember_correct_rate": 0.0,
            "not_remember_rate": 0.0,
            "distractor_irrelevant_rate": 0.0,
            "other_rate": 0.0,
        }

    def _rate(answer_type: str) -> float:
        return sum(1 for item in items if item.get("predicted_answer_type") == answer_type) / total

    other_rate = sum(
        1
        for item in items
        if item.get("predicted_answer_type") not in {"remember_correct", "not_remember", "distractor_irrelevant"}
    ) / total
    return {
        "num_questions": total,
        "remember_correct_rate": _rate("remember_correct"),
        "not_remember_rate": _rate("not_remember"),
        "distractor_irrelevant_rate": _rate("distractor_irrelevant"),
        "other_rate": other_rate,
    }


def _load_sidecar(rendered: Dict[str, Any], explicit_sidecar: str) -> Dict[str, Any]:
    sidecar_path = explicit_sidecar or rendered.get("source_sidecar", "")
    if not sidecar_path:
        return {}
    path = Path(sidecar_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_world_transform(conversation: Dict[str, Any], sidecar: Dict[str, Any], world: str) -> Dict[str, Any]:
    if world == "baseline":
        return conversation

    transformed = conversation
    if world == "no_store":
        for turn in sidecar.get("key_turns", []):
            timestamp = turn.get("timestamp")
            if not timestamp:
                continue
            transformed = apply_no_store(
                transformed,
                period="Conversation Initial Stage",
                key_timestamp=timestamp,
            )
        return transformed

    if world == "forget":
        return apply_forget(transformed, instruction_period="Conversation Early Stage")

    if world == "no_use":
        return apply_no_use(
            transformed,
            restrict_period="Conversation Early Stage",
            release_period=None,
        )

    raise ValueError(f"Unsupported world: {world}")


def _format_amem_memories(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No relevant memories were retrieved."
    lines = []
    for idx, item in enumerate(results, start=1):
        content = item.get("content", "")
        context = item.get("context", "")
        tags = item.get("tags", [])
        score = item.get("score")
        score_str = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
        lines.append(f"{idx}. {content}{score_str} | context={context} | tags={tags}")
    return "\n".join(lines)


def _load_amem_system(model: str, embedding_model: str, api_key_file: str = "openrouter_key.txt"):
    try:
        from agentic_memory.memory_system import AgenticMemorySystem
        from agentic_memory.llm_controller import OpenAIController
    except ImportError as exc:
        raise ImportError(
            "A-Mem is not installed in the current environment. Install the agiresearch/A-mem package first."
        ) from exc

    _patch_amem_openai_controller(OpenAIController)
    _ensure_openrouter_env(api_key_file)
    api_key = _load_api_key(api_key_file)
    memory_system = AgenticMemorySystem(
        model_name=embedding_model,
        llm_backend="openai",
        llm_model=_resolve_model_name(model),
        api_key=api_key,
    )
    return memory_system


def _patch_amem_openai_controller(controller_cls: Any) -> None:
    if getattr(controller_cls, "_gpt54mini_patch_applied", False):
        return

    def _patched_get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
            "temperature": temperature,
        }
        if "gpt-5" in str(self.model):
            request_kwargs["max_completion_tokens"] = 1000
        else:
            request_kwargs["max_tokens"] = 1000
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content

    controller_cls.get_completion = _patched_get_completion
    controller_cls._gpt54mini_patch_applied = True


class AMemRetrievalAgent:
    def __init__(
        self,
        memory_system: Any,
        client: OpenAI,
        model: str,
        memory_limit: int,
        persona_messages: List[Dict[str, str]],
    ) -> None:
        self.memory_system = memory_system
        self.client = client
        self.model = model
        self.memory_limit = memory_limit
        self.persona_messages = persona_messages

    def preload(self, context_messages: List[Dict[str, str]]) -> None:
        self.memory_system.memories = {}
        try:
            self.memory_system.retriever.client.reset()
            self.memory_system.retriever = type(self.memory_system.retriever)(
                collection_name="memories",
                model_name=self.memory_system.model_name,
            )
        except Exception:
            pass

        for idx, msg in enumerate(context_messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            note_content = f"{role}: {content}"
            self.memory_system.add_note(
                content=note_content,
                category="conversation",
                tags=[role],
                time=f"2025010100{idx:02d}",
            )

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        retrieved = self.memory_system.search_agentic(question, k=self.memory_limit)
        prompt = _build_eval_prompt(question, choices)
        memories_text = _format_amem_memories(retrieved)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": f"Retrieved memories:\n{memories_text}\n\n{prompt}",
            },
        ]
        response = _request_text(self.client, self.model, messages)
        return {
            "model_response": response,
            "retrieved_memories": retrieved,
        }


def _score_item(
    agent: Any,
    question: str,
    choices: Dict[str, str],
    choice_to_answer_type: Dict[str, str],
) -> Dict[str, Any]:
    labels = list(choices.keys())
    result = agent.answer_mcq(question, choices)
    predicted_choice = _extract_choice(result.get("model_response", ""), labels)
    predicted_type = choice_to_answer_type.get(predicted_choice, "")
    return {
        "choices": choices,
        "choice_to_answer_type": choice_to_answer_type,
        "model_response": result.get("model_response", ""),
        "predicted_choice": predicted_choice,
        "predicted_answer_type": predicted_type,
        "retrieved_memories": result.get("retrieved_memories"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recall MCQs with an A-Mem-backed agent.")
    parser.add_argument(
        "--rendered",
        default="data/test/travelPlanning/specs/conversation_travelPlanning_persona0_sample0.recall_rendered.json",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--ask_period", default="Conversation Late Stage")
    parser.add_argument("--world", choices=["baseline", "no_store", "forget", "no_use"], default="baseline")
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--memory_limit", type=int, default=5)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rendered = json.loads(Path(args.rendered).read_text(encoding="utf-8"))
    conversation_path = rendered["source_conversation"]
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = _load_sidecar(rendered, args.sidecar)
    transformed_conversation = _apply_world_transform(conversation, sidecar, args.world)
    persona_messages = _build_persona_system_message(transformed_conversation)
    context_messages = persona_messages + build_context_messages(transformed_conversation, args.ask_period)

    client = _load_openai_client()
    memory_system = _load_amem_system(args.model, args.embedding_model)
    agent = AMemRetrievalAgent(
        memory_system,
        client=client,
        model=args.model,
        memory_limit=args.memory_limit,
        persona_messages=persona_messages,
    )
    agent.preload(context_messages)

    results = {
        "source_rendered": args.rendered,
        "source_conversation": conversation_path,
        "source_sidecar": rendered.get("source_sidecar", args.sidecar),
        "model": args.model,
        "backend": "a_mem_retrieval",
        "world": args.world,
        "ask_period": args.ask_period,
        "memory_limit": args.memory_limit,
        "whole_recall_results": [],
        "slot_recall_results": [],
    }

    for item in rendered.get("whole_recall_set", []):
        if not _is_valid_mcq(
            item["rendered"]["choices"],
            item["rendered"]["choice_to_answer_type"],
        ):
            continue
        scored = _score_item(
            agent,
            item["rendered"]["question"],
            item["rendered"]["choices"],
            item["rendered"]["choice_to_answer_type"],
        )
        results["whole_recall_results"].append(
            {
                "timestamp": item["timestamp"],
                "turn_role": item["turn_role"],
                "identifier_label": item["identifier_label"],
                "question": item["rendered"]["question"],
                **scored,
            }
        )

    for item in rendered.get("slot_recall_set", []):
        for slot_item in item["rendered"].get("items", []):
            if not _is_valid_mcq(
                slot_item["choices"],
                slot_item["choice_to_answer_type"],
            ):
                continue
            scored = _score_item(
                agent,
                slot_item["question"],
                slot_item["choices"],
                slot_item["choice_to_answer_type"],
            )
            results["slot_recall_results"].append(
                {
                    "timestamp": item["timestamp"],
                    "turn_role": item["turn_role"],
                    "identifier_label": item["identifier_label"],
                    "sensitive_key": slot_item["sensitive_key"],
                    "sensitive_value": slot_item["sensitive_value"],
                    "question": slot_item["question"],
                    **scored,
                }
            )

    results["summary"] = {
        "whole_recall": _rate_summary(results["whole_recall_results"]),
        "slot_recall": _rate_summary(results["slot_recall_results"]),
        "key_turns": _rate_summary(
            [item for item in results["whole_recall_results"] if item.get("turn_role") == "key"]
            + [item for item in results["slot_recall_results"] if item.get("turn_role") == "key"]
        ),
        "probe_turns": _rate_summary(
            [item for item in results["whole_recall_results"] if item.get("turn_role") == "probe"]
            + [item for item in results["slot_recall_results"] if item.get("turn_role") == "probe"]
        ),
    }

    suffix = f".{args.world}.a_mem_retrieval_eval_{args.model}.json"
    if args.ask_period != "Conversation Late Stage":
        suffix = f".{args.world}.{_ask_period_tag(args.ask_period)}.a_mem_retrieval_eval_{args.model}.json"
    output_path = args.output or args.rendered.replace(".recall_rendered.json", suffix)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
