import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

SlotKey = Tuple[str, str, str, str]


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_credentials(api_key_file: str = "keys/openrouter_key.txt") -> Tuple[str, str]:
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
    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or OPENROUTER_BASE_URL
    return api_key, base_url


def _load_client(api_key_file: str = "keys/openrouter_key.txt") -> OpenAI:
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


def _slot_key(item: Dict[str, Any]) -> SlotKey:
    return (
        str(item.get("identifier_label", "")).strip(),
        str(item.get("question", "")).strip(),
        str(item.get("sensitive_key", "")).strip(),
        str(item.get("sensitive_value", "")).strip(),
    )


def _build_slot_batch_prompt(batch: List[Tuple[int, Dict[str, str]]]) -> str:
    allowed = ", ".join(ALLOWED_SLOT_TYPES)
    payload = [
        {
            "id": idx,
            "identifier_label": item["identifier_label"],
            "question": item["question"],
            "sensitive_key": item["sensitive_key"],
            "sensitive_value": item["sensitive_value"],
        }
        for idx, item in batch
    ]
    return (
        "You are assigning coarse slot types for memory-analysis.\n"
        f"Choose exactly one label from: {allowed}.\n"
        "Return valid JSON only as a list of objects with keys `id`, `slot_type_llm`, and `reason`.\n"
        "The label should describe the kind of detail the slot is about, not whether it was remembered.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
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


def classify_slot_item(client: OpenAI, model: str, item: Dict[str, Any]) -> Tuple[str, str]:
    return _classify_item(client, model, item)


def _load_cache(cache_path: Path) -> Dict[SlotKey, Dict[str, str]]:
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    out: Dict[SlotKey, Dict[str, str]] = {}
    for entry in raw:
        key = (
            entry["identifier_label"],
            entry["question"],
            entry["sensitive_key"],
            entry["sensitive_value"],
        )
        out[key] = {
            "slot_type_llm": entry["slot_type_llm"],
            "slot_type_llm_reason": entry.get("slot_type_llm_reason", ""),
        }
    return out


def _save_cache(cache_path: Path, cache: Dict[SlotKey, Dict[str, str]]) -> None:
    rows = []
    for key, value in sorted(cache.items()):
        rows.append(
            {
                "identifier_label": key[0],
                "question": key[1],
                "sensitive_key": key[2],
                "sensitive_value": key[3],
                "slot_type_llm": value["slot_type_llm"],
                "slot_type_llm_reason": value.get("slot_type_llm_reason", ""),
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _classify_batched(
    client: OpenAI,
    keyed_items: Dict[SlotKey, Dict[str, str]],
    model: str,
    batch_size: int,
    max_retries: int,
    cache: Dict[SlotKey, Dict[str, str]],
    cache_path: Optional[Path],
) -> Dict[SlotKey, Dict[str, str]]:
    keys = list(keyed_items.keys())
    out: Dict[SlotKey, Dict[str, str]] = {}
    for start in range(0, len(keys), batch_size):
        chunk_keys = keys[start : start + batch_size]
        batch = [(idx, keyed_items[key]) for idx, key in enumerate(chunk_keys)]
        parsed = None
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                raw = _request_text(client, model, [{"role": "user", "content": _build_slot_batch_prompt(batch)}])
                parsed = json.loads(raw)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(min(20, 2 ** attempt))
        if not isinstance(parsed, list):
            for key in chunk_keys:
                label, reason = _classify_item(client, model, keyed_items[key])
                out[key] = {"slot_type_llm": label, "slot_type_llm_reason": reason}
                cache[key] = out[key]
            if cache_path is not None:
                _save_cache(cache_path, cache)
            if last_error is not None:
                print(f"batch fallback at {start}: {last_error}", flush=True)
            continue

        by_id = {
            int(entry["id"]): (
                str(entry.get("slot_type_llm", "")).strip(),
                str(entry.get("reason", "")).strip(),
            )
            for entry in parsed
            if "id" in entry
        }
        for idx, key in enumerate(chunk_keys):
            label, reason = by_id.get(idx, ("other_detail", "missing_in_batch_response"))
            if label not in ALLOWED_SLOT_TYPES:
                label = "other_detail"
            out[key] = {"slot_type_llm": label, "slot_type_llm_reason": reason}
            cache[key] = out[key]
        if cache_path is not None:
            _save_cache(cache_path, cache)
        print(f"annotated {min(start + batch_size, len(keys))}/{len(keys)} unique slots", flush=True)
    return out


def annotate_slot_items(
    items: List[Dict[str, Any]],
    model: str = "gpt-5-mini",
    api_key_file: str = "keys/openrouter_key.txt",
    batch_size: int = 25,
    cache_path: str = "",
    max_retries: int = 4,
) -> None:
    client = _load_client(api_key_file)
    label_cache: Dict[SlotKey, Dict[str, str]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None:
        label_cache.update(_load_cache(cache_file))

    keyed_items: Dict[SlotKey, Dict[str, str]] = {}
    for item in items:
        key = _slot_key(item)
        if key not in label_cache:
            keyed_items[key] = {
                "identifier_label": key[0],
                "question": key[1],
                "sensitive_key": key[2],
                "sensitive_value": key[3],
            }

    if keyed_items:
        if batch_size > 1:
            _classify_batched(
                client,
                keyed_items,
                model,
                batch_size,
                max_retries,
                label_cache,
                cache_file,
            )
        else:
            for key, payload in keyed_items.items():
                label, reason = _classify_item(client, model, payload)
                label_cache[key] = {"slot_type_llm": label, "slot_type_llm_reason": reason}
            if cache_file is not None:
                _save_cache(cache_file, label_cache)

    for item in items:
        key = _slot_key(item)
        label_info = label_cache.get(key, {"slot_type_llm": "other_detail", "slot_type_llm_reason": "missing_label"})
        slot_type_llm = label_info["slot_type_llm"]
        reason = label_info.get("slot_type_llm_reason", "")
        item["slot_type_llm"] = slot_type_llm
        item["slot_type_llm_reason"] = reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate slot recall results with an LLM-derived slot type.")
    parser.add_argument("--input", default="", help="Path to an eval JSON file.")
    parser.add_argument("--root", default="", help="Optional root directory of eval JSON files to annotate in place.")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to overwriting the input file.")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api_key_file", default="keys/openrouter_key.txt")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--cache", default="", help="Optional JSON cache path for slot labels.")
    args = parser.parse_args()

    if bool(args.input) == bool(args.root):
        raise SystemExit("Provide exactly one of --input or --root.")

    if args.root:
        root = Path(args.root)
        paths: List[Path] = []
        for path in root.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data.get("slot_recall_results"), list):
                paths.append(path)
        touched = 0
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            annotate_slot_items(
                data.get("slot_recall_results", []),
                model=args.model,
                api_key_file=args.api_key_file,
                batch_size=args.batch_size,
                cache_path=args.cache,
            )
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            touched += 1
        print(f"updated {touched} files")
        return

    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    annotate_slot_items(
        data.get("slot_recall_results", []),
        model=args.model,
        api_key_file=args.api_key_file,
        batch_size=args.batch_size,
        cache_path=args.cache,
    )

    output_path = Path(args.output) if args.output else path
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
