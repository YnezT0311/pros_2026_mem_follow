import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..common import period_tag
from ..transforms import apply_no_store, apply_no_use, apply_staged_forget


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "MemoryCtrl"
DEFAULT_MODEL_ALIASES = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
}


def resolve_model_name(model: str, aliases: Optional[Dict[str, str]] = None) -> str:
    merged_aliases = dict(DEFAULT_MODEL_ALIASES)
    if aliases:
        merged_aliases.update(aliases)
    return merged_aliases.get(model, model)


def load_openai_credentials(
    api_key_file: str = "keys/openrouter_key.txt",
    *,
    default_base_url: str = OPENROUTER_BASE_URL,
) -> tuple[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and Path(api_key_file).exists():
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        legacy_key = os.getenv("OPENAI_API_KEY", "").strip()
        legacy_path = Path("keys/openai_key.txt")
        if not legacy_key and legacy_path.exists():
            legacy_key = legacy_path.read_text(encoding="utf-8").strip()
        api_key = legacy_key
    if not api_key:
        raise FileNotFoundError("No API key found. Set OPENROUTER_API_KEY or provide keys/openrouter_key.txt.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or default_base_url
    return api_key, base_url


def load_openai_client(
    api_key_file: str = "keys/openrouter_key.txt",
    *,
    title: str = OPENROUTER_TITLE,
    default_base_url: str = OPENROUTER_BASE_URL,
) -> OpenAI:
    api_key, base_url = load_openai_credentials(api_key_file, default_base_url=default_base_url)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-OpenRouter-Title": title},
    )


def ensure_openai_env(
    api_key_file: str = "keys/openrouter_key.txt",
    *,
    default_base_url: str = OPENROUTER_BASE_URL,
) -> None:
    api_key, base_url = load_openai_credentials(api_key_file, default_base_url=default_base_url)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url


def extract_text(resp: Any) -> str:
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


def request_text(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    *,
    aliases: Optional[Dict[str, str]] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    resolved_model = resolve_model_name(model, aliases=aliases)

    response_kwargs: Dict[str, Any] = {"model": resolved_model, "input": messages}
    if temperature is not None:
        response_kwargs["temperature"] = temperature
    if timeout is not None:
        response_kwargs["timeout"] = timeout
    if reasoning_effort:
        response_kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        resp = client.responses.create(**response_kwargs)
        text = extract_text(resp)
        if text:
            return text
    except Exception:
        pass

    chat_kwargs: Dict[str, Any] = {"model": resolved_model, "messages": messages}
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if timeout is not None:
        chat_kwargs["timeout"] = timeout
    if reasoning_effort:
        chat_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    completion = client.chat.completions.create(**chat_kwargs)
    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Reasoning models (e.g. Gemini 3.x Pro) sometimes return content=None when
    # the answer leaked into the reasoning trace. Fall back so the eval can
    # still parse a choice from the reasoning text.
    for fallback_attr in ("reasoning", "reasoning_content"):
        fallback = getattr(message, fallback_attr, None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return ""


def mark_cache_breakpoint(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag the last message with a `cache_control: ephemeral` breakpoint.

    OpenRouter caches everything from the start of the request up to and
    including the marked block. Providers that don't support explicit
    caching (e.g. OpenAI, which caches implicitly) ignore the marker.
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        new_content = list(content)
        tail = dict(new_content[-1])
        tail["cache_control"] = {"type": "ephemeral"}
        new_content[-1] = tail
        last["content"] = new_content
    out[-1] = last
    return out


def build_eval_prompt(question: str, choices: Dict[str, str]) -> str:
    labels = list(choices.keys())
    rendered_options = [f"({label.lower()}) {choices[label]}" for label in labels]
    instructions = (
        "Find the most appropriate model response and give your final answer "
        f"{', '.join(f'({label.lower()})' for label in labels)} after the special token <final_answer>."
    )
    return f"Question: {question}\n\n{instructions}\n\n{rendered_options}"


def extract_choice(text: str, labels: List[str]) -> str:
    cleaned = text.strip()
    labels_upper = [label.upper() for label in labels]

    final_answer_patterns = [
        r"<final_answer>\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?",
        r"<final_answer>\s*[:\-]?\s*([A-Za-z])\b",
    ]
    for pattern in final_answer_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in labels_upper:
                return candidate

    option_marker_matches = re.findall(r"\(([A-Za-z])\)", cleaned, flags=re.IGNORECASE)
    for candidate in reversed(option_marker_matches):
        candidate = candidate.upper()
        if candidate in labels_upper:
            return candidate

    cleaned = cleaned.upper()
    for label in labels:
        pattern = rf"\b({re.escape(label.upper())})\b"
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return ""


def build_persona_system_message(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    persona = conversation.get("Expanded Persona")
    if not isinstance(persona, str) or not persona.strip():
        return []
    return [{"role": "system", "content": f"Current user persona: {persona.strip()}"}]


def ask_period_tag(ask_period: str) -> str:
    return period_tag(ask_period)


def build_label_map(rendered: Dict[str, Any]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for item in rendered.get("whole_recall_set", []):
        timestamp = str(item.get("timestamp", "")).strip()
        label = str(item.get("identifier_label", "")).strip()
        if timestamp and label:
            label_map[timestamp] = label
    return label_map


def is_valid_mcq(choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> bool:
    return bool(choices) and bool(choice_to_answer_type)


def load_sidecar(rendered: Dict[str, Any], explicit_sidecar: str) -> Dict[str, Any]:
    sidecar_path = explicit_sidecar or rendered.get("source_sidecar", "")
    if not sidecar_path:
        return {}
    path = Path(sidecar_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_world_transform(
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
