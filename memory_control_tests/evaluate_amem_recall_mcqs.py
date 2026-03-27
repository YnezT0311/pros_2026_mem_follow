import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from .transforms import apply_forget, apply_no_store, apply_no_use, build_context_messages


def _load_api_key(api_key_file: str = "openai_key.txt") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and Path(api_key_file).exists():
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise FileNotFoundError("No API key found. Set OPENAI_API_KEY or provide openai_key.txt.")
    return api_key


def _load_openai_client(api_key_file: str = "openai_key.txt") -> OpenAI:
    api_key = _load_api_key(api_key_file)
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


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
    lines = [
        "Answer the following multiple-choice question using the retrieved persistent memories when relevant.",
        f"Choose the single best answer and reply with only one of: {', '.join(labels)}.",
        f"Question: {question}",
    ]
    for label in labels:
        lines.append(f"{label}. {choices[label]}")
    return "\n".join(lines)


def _extract_choice(text: str, labels: List[str]) -> str:
    cleaned = text.strip().upper()
    for label in labels:
        pattern = rf"\b({re.escape(label.upper())})\b"
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return ""


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


def _load_amem_system(model: str, embedding_model: str, api_key_file: str = "openai_key.txt"):
    try:
        from agentic_memory.memory_system import AgenticMemorySystem
    except ImportError as exc:
        raise ImportError(
            "A-Mem is not installed in the current environment. Install the agiresearch/A-mem package first."
        ) from exc

    api_key = _load_api_key(api_key_file)
    memory_system = AgenticMemorySystem(
        model_name=embedding_model,
        llm_backend="openai",
        llm_model=model,
        evo_threshold=10**9,
        api_key=api_key,
    )

    # Avoid expensive evolution-time LLM calls during benchmark preload.
    memory_system.process_memory = lambda note: (False, note)
    return memory_system


class AMemRetrievalAgent:
    def __init__(self, memory_system: Any, client: OpenAI, model: str, memory_limit: int) -> None:
        self.memory_system = memory_system
        self.client = client
        self.model = model
        self.memory_limit = memory_limit

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
        labels = list(choices.keys())
        retrieved = self.memory_system.search_agentic(question, k=self.memory_limit)
        prompt = _build_eval_prompt(question, choices)
        memories_text = _format_amem_memories(retrieved)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant using retrieved persistent memories. "
                    "Use the retrieved memories when they are relevant, then answer the multiple-choice question. "
                    f"Reply with only one choice label from: {', '.join(labels)}."
                ),
            },
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
        default="data/baseline/travelPlanning/conversation_travelPlanning_persona0_sample0.recall_rendered.json",
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
    context_messages = build_context_messages(transformed_conversation, args.ask_period)

    client = _load_openai_client()
    memory_system = _load_amem_system(args.model, args.embedding_model)
    agent = AMemRetrievalAgent(memory_system, client=client, model=args.model, memory_limit=args.memory_limit)
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

    output_path = args.output or args.rendered.replace(
        ".recall_rendered.json",
        f".{args.world}.a_mem_retrieval_eval_{args.model}.json",
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
