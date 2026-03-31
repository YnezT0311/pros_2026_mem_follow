import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "MemoryCtrl"
MODEL_ALIASES = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-5.1": "openai/gpt-5.1",
}

ALLOWED_SLOT_TYPES = [
    "budget",
    "date_or_time",
    "email",
    "phone",
    "location_or_contact_point",
    "medical_or_access_need",
    "dietary_requirement",
    "document_or_account_reference",
    "preference_or_requirement",
    "other_detail",
]


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_credentials(api_key_file: str = "openrouter_key.txt") -> Tuple[str, str]:
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


def _load_client(api_key_file: str = "openrouter_key.txt") -> OpenAI:
    api_key, base_url = _load_credentials(api_key_file)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-OpenRouter-Title": OPENROUTER_TITLE},
    )


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
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
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content.strip()


def _build_prompt(item: Dict[str, Any]) -> str:
    allowed = ", ".join(ALLOWED_SLOT_TYPES)
    payload = {
        "identifier_label": item.get("identifier_label", ""),
        "question": item.get("question", ""),
        "sensitive_key": item.get("sensitive_key", ""),
        "sensitive_value": item.get("sensitive_value", ""),
    }
    return (
        "You are assigning a coarse slot type for memory-analysis.\n"
        f"Choose exactly one label from: {allowed}.\n"
        "Return valid JSON only with keys `slot_type_llm` and `reason`.\n"
        "The label should describe what kind of detail the slot is about, not whether it was remembered.\n\n"
        f"Item:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _classify_item(client: OpenAI, model: str, item: Dict[str, Any]) -> Tuple[str, str]:
    prompt = _build_prompt(item)
    raw = _request_text(client, model, [{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(raw)
    except Exception:
        cleaned = raw.strip().strip("`").strip()
        label = cleaned.splitlines()[0].strip().strip('"')
        if label not in ALLOWED_SLOT_TYPES:
            label = "other_detail"
        return label, "fallback_parse"
    label = str(parsed.get("slot_type_llm", "")).strip()
    reason = str(parsed.get("reason", "")).strip()
    if label not in ALLOWED_SLOT_TYPES:
        label = "other_detail"
    return label, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate slot recall results with an LLM-derived slot type.")
    parser.add_argument("--input", required=True, help="Path to an eval JSON file.")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to overwriting the input file.")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api_key_file", default="openrouter_key.txt")
    args = parser.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    client = _load_client(args.api_key_file)

    cache: Dict[Tuple[str, str, str, str], Tuple[str, str]] = {}
    for item in data.get("slot_recall_results", []):
        key = (
            str(item.get("identifier_label", "")).strip(),
            str(item.get("question", "")).strip(),
            str(item.get("sensitive_key", "")).strip(),
            str(item.get("sensitive_value", "")).strip(),
        )
        if key not in cache:
            cache[key] = _classify_item(client, args.model, item)
        slot_type_llm, reason = cache[key]
        item["slot_type_llm"] = slot_type_llm
        item["slot_type_llm_reason"] = reason

    output_path = Path(args.output) if args.output else path
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
