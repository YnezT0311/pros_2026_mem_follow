import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from openai import OpenAI

from ..common import build_recall_summary, build_transformed_history_path, period_tag, rewrite_key_references
from ..transforms import apply_no_store, apply_staged_forget, apply_no_use, build_context_messages


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "MemoryCtrl"
MODEL_ALIASES = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
}


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_openai_credentials(api_key_file: str = "openrouter_key.txt") -> tuple[str, str]:
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
    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or OPENROUTER_BASE_URL
    return api_key, base_url


def _load_openai_client(api_key_file: str = "openrouter_key.txt") -> OpenAI:
    api_key, base_url = _load_openai_credentials(api_key_file)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-OpenRouter-Title": OPENROUTER_TITLE},
    )


def _load_optional_text_file(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def _load_first_available_file(paths: List[str]) -> str:
    for path in paths:
        value = _load_optional_text_file(path)
        if value:
            return value
    return ""


def _load_zep_client(
    api_key_file: str = "zep_api_key.txt",
    api_url_file: str = "zep_api_url.txt",
):
    try:
        from zep_cloud import Zep
    except ImportError as exc:
        raise ImportError(
            "Zep SDK is not installed in the current environment. Use the zep311 environment to run this evaluator."
        ) from exc

    api_key = os.getenv("ZEP_API_KEY", "").strip() or _load_first_available_file(
        [api_key_file, "zep_api-key.txt"]
    )
    if not api_key:
        raise FileNotFoundError(
            "No ZEP_API_KEY found. Set ZEP_API_KEY or provide zep_api_key.txt to use the Zep evaluator."
        )

    api_url = os.getenv("ZEP_API_URL", "").strip() or _load_first_available_file(
        [api_url_file, "zep_api-url.txt"]
    )
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_url:
        kwargs["base_url"] = f"{api_url.rstrip('/')}/api/v2"
    return Zep(**kwargs)


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
    return period_tag(ask_period)


def _build_label_map(rendered: Dict[str, Any]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for item in rendered.get("whole_recall_set", []):
        timestamp = str(item.get("timestamp", "")).strip()
        label = str(item.get("identifier_label", "")).strip()
        if timestamp and label:
            label_map[timestamp] = label
    return label_map


def _is_valid_mcq(choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> bool:
    return bool(choices) and bool(choice_to_answer_type)


def _load_sidecar(rendered: Dict[str, Any], explicit_sidecar: str) -> Dict[str, Any]:
    sidecar_path = explicit_sidecar or rendered.get("source_sidecar", "")
    if not sidecar_path:
        return {}
    path = Path(sidecar_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_world_transform(
    conversation: Dict[str, Any],
    sidecar: Dict[str, Any],
    world: str,
    target_references: List[str],
    no_use_restrict_period: str,
    no_use_release_period: str,
) -> Dict[str, Any]:
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
        return apply_staged_forget(
            transformed,
            target_references=target_references,
        )

    if world == "no_use":
        return apply_no_use(
            transformed,
            restrict_period=no_use_restrict_period,
            release_period=(no_use_release_period or None),
        )

    raise ValueError(f"Unsupported world: {world}")


def _format_zep_context(context: str) -> str:
    cleaned = (context or "").strip()
    if not cleaned:
        return "No relevant memories were retrieved."
    return cleaned


def _safe_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "item"


class ZepRetrievalAgent:
    MAX_HISTORY_MESSAGES = 29

    def __init__(
        self,
        zep_client: Any,
        openai_client: OpenAI,
        model: str,
        user_id: str,
        persona_messages: List[Dict[str, str]],
        context_messages: List[Dict[str, str]],
    ) -> None:
        from zep_cloud import Message

        self.zep_client = zep_client
        self.openai_client = openai_client
        self.model = model
        self.user_id = _safe_token(user_id)
        self.persona_messages = persona_messages
        self.context_messages = context_messages
        self.message_cls = Message
        self.thread_id = f"{self.user_id}_{uuid4().hex}"
        self._prepared = False

    def _ensure_user(self) -> None:
        try:
            self.zep_client.user.get(self.user_id)
            return
        except Exception:
            pass
        self.zep_client.user.add(
            user_id=self.user_id,
            metadata={"source": "personamem_eval"},
        )

    def _thread_messages(self) -> List[Any]:
        messages: List[Any] = []
        for idx, msg in enumerate(self.context_messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            messages.append(
                self.message_cls(
                    role=role,
                    content=content,
                    metadata={"position": idx},
                )
            )
        return messages[-self.MAX_HISTORY_MESSAGES :]

    def prepare(self) -> None:
        if self._prepared:
            return
        self._ensure_user()
        self.zep_client.thread.create(thread_id=self.thread_id, user_id=self.user_id)
        history_messages = self._thread_messages()
        if history_messages:
            self.zep_client.thread.add_messages(self.thread_id, messages=history_messages)
        self._prepared = True

    def close(self) -> None:
        try:
            self.zep_client.thread.delete(self.thread_id)
        except Exception:
            pass

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        self.prepare()
        context = ""
        try:
            question_response = self.zep_client.thread.add_messages(
                self.thread_id,
                messages=[self.message_cls(role="user", content=question)],
                return_context=True,
            )
            context = getattr(question_response, "context", "") or ""
            if not context:
                context = getattr(self.zep_client.thread.get_user_context(self.thread_id), "context", "") or ""
        except Exception:
            raise

        prompt = _build_eval_prompt(question, choices)
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": f"Retrieved memories:\n{_format_zep_context(context)}\n\n{prompt}",
            },
        ]
        response = _request_text(self.openai_client, self.model, messages)
        return {
            "model_response": response,
            "retrieved_memories": {"context": context},
        }


def _score_item(
    agent: ZepRetrievalAgent,
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
    parser = argparse.ArgumentParser(description="Evaluate recall MCQs with a Zep-backed retrieval agent.")
    parser.add_argument(
        "--rendered",
        default="data/test/travelPlanning/specs/conversation_travelPlanning_persona0_sample0.recall_rendered.json",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--ask_period", default="Conversation Late Stage")
    parser.add_argument("--world", choices=["baseline", "no_store", "forget", "no_use"], default="baseline")
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--api_key_file", default="openrouter_key.txt")
    parser.add_argument("--zep_api_key_file", default="zep_api_key.txt")
    parser.add_argument("--zep_api_url_file", default="zep_api_url.txt")
    parser.add_argument("--no_use_restrict_period", default="Conversation Early Stage")
    parser.add_argument("--no_use_release_period", default="")
    args = parser.parse_args()

    rendered = json.loads(Path(args.rendered).read_text(encoding="utf-8"))
    conversation_path = rendered["source_conversation"]
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = _load_sidecar(rendered, args.sidecar)
    label_map = _build_label_map(rendered)
    rewrite_client = _load_openai_client(args.api_key_file)
    transformed_history_path = build_transformed_history_path(
        args.rendered,
        args.world,
        release_period=args.no_use_release_period or None,
        restrict_period=args.no_use_restrict_period,
    )
    if transformed_history_path.exists():
        transformed_conversation = json.loads(transformed_history_path.read_text(encoding="utf-8"))
    else:
        target_references = rewrite_key_references(
            lambda model, prompt: _request_text(rewrite_client, model, [{"role": "user", "content": prompt}]),
            args.model,
            sidecar.get("key_turns", [])[:3],
            label_map=label_map,
        )
        transformed_conversation = _apply_world_transform(
            conversation,
            sidecar,
            args.world,
            target_references,
            args.no_use_restrict_period,
            args.no_use_release_period,
        )
        transformed_history_path.write_text(json.dumps(transformed_conversation, ensure_ascii=False, indent=2), encoding="utf-8")
    persona_messages = _build_persona_system_message(transformed_conversation)
    context_messages = build_context_messages(transformed_conversation, args.ask_period)

    openai_client = _load_openai_client(args.api_key_file)
    zep_client = _load_zep_client(args.zep_api_key_file, args.zep_api_url_file)
    user_id = Path(args.rendered).stem
    agent = ZepRetrievalAgent(
        zep_client=zep_client,
        openai_client=openai_client,
        model=args.model,
        user_id=user_id,
        persona_messages=persona_messages,
        context_messages=context_messages,
    )

    results = {
        "source_rendered": args.rendered,
        "source_conversation": conversation_path,
        "source_sidecar": rendered.get("source_sidecar", args.sidecar),
        "model": args.model,
        "backend": "zep_retrieval",
        "world": args.world,
        "ask_period": args.ask_period,
        "no_use_restrict_period": args.no_use_restrict_period,
        "no_use_release_period": args.no_use_release_period,
        "whole_recall_results": [],
        "slot_recall_results": [],
    }

    try:
        agent.prepare()

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
    finally:
        agent.close()

    results["summary"] = build_recall_summary(
        args.world,
        results["whole_recall_results"],
        results["slot_recall_results"],
    )

    if args.world == "no_use":
        suffix = f".{args.world}.restrict_{_ask_period_tag(args.no_use_restrict_period)}"
        if args.no_use_release_period:
            suffix += f".release_{_ask_period_tag(args.no_use_release_period)}"
        suffix += f".test_{_ask_period_tag(args.ask_period)}.zep_retrieval_eval_{args.model}.json"
    else:
        suffix = f".{args.world}.zep_retrieval_eval_{args.model}.json"
        if args.ask_period != "Conversation Late Stage":
            suffix = f".{args.world}.{_ask_period_tag(args.ask_period)}.zep_retrieval_eval_{args.model}.json"
    output_path = args.output or args.rendered.replace(".recall_rendered.json", suffix)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results["transformed_history_path"] = str(transformed_history_path)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
