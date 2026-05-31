#!/usr/bin/env python3
"""Generate natural application forget references.

This mirrors the recall benchmark's reference-rewrite intent: each target turn
gets a short noun phrase that a user could naturally use later to identify what
should be forgotten. The generated phrase is written to each source
`application_items.json` as `forget_reference`, so world rebuilds are stable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_control_tests.evaluation.shared import load_openai_client, request_text


DEFAULT_TOPICS = ["travelPlanning", "financialConsultation", "medicalConsultation"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def item_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise TypeError("application items file must be a list or object with items")


def build_prompt(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    targets = []
    for item in items:
        targets.append(
            {
                "id": item["id"],
                "target_user_turn": item.get("target_user_turn", ""),
                "memory_to_reference": item.get("unique_long_term_info") or item.get("lasting_memory", ""),
                "application_question": item.get("question", ""),
            }
        )
    system = (
        "You write short natural references for memory-control forget instructions. "
        "Return only JSON."
    )
    user = (
        "For each target below, write a short bare noun phrase that a user could use later "
        "to refer to exactly what they want forgotten.\n\n"
        "Rules:\n"
        "- Return a JSON object mapping each id to one phrase.\n"
        "- Each phrase should be 3-12 words when possible.\n"
        "- It should point to the target turn's memorable request/detail, not to metadata.\n"
        "- Do not write a full sentence.\n"
        "- Do not use wrappers like 'what I shared about', 'my earlier request about', or 'that earlier preference'.\n"
        "- Do not say 'the user'.\n"
        "- Prefer natural phrases like 'the Tokyo seafood allergy reservation note', "
        "'the spreadsheet-based budget review setup', or 'the morning follow-up appointment constraint'.\n\n"
        f"Targets:\n{json.dumps(targets, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict[str, str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("model response was not a JSON object")
    return {str(k): " ".join(str(v).strip().split()) for k, v in data.items()}


def valid_reference(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    lowered = text.lower()
    if len(words) > 16:
        return False
    banned = [
        "the user",
        "what i shared",
        "what i told",
        "my earlier request",
        "that earlier preference",
        "unique_long_term_info",
    ]
    return not any(bad in lowered for bad in banned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-root", default="data/application/mcq_work")
    parser.add_argument("--sample", default="persona0_sample0")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-key-file", default="keys/openrouter_key.txt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Keep existing valid forget_reference values when model generation is unavailable or invalid.",
    )
    args = parser.parse_args()

    files: list[tuple[Path, Any, list[dict[str, Any]]]] = []
    kept: list[dict[str, Any]] = []
    for topic in args.topics:
        path = Path(args.items_root) / topic / args.sample / "application_items.json"
        data = load_json(path)
        items = item_rows(data)
        files.append((path, data, items))
        kept.extend(item for item in items if item.get("status") == "kept")

    client = load_openai_client(args.api_key_file)
    raw = request_text(
        client,
        args.model,
        build_prompt(kept),
        temperature=0,
        timeout=180,
    )
    generated = parse_json_object(raw)

    invalid = []
    for item in kept:
        ref = generated.get(str(item["id"]), "")
        if not valid_reference(ref):
            if args.preserve_existing and valid_reference(str(item.get("forget_reference") or "").strip()):
                ref = str(item.get("forget_reference") or "").strip()
            else:
                invalid.append(item["id"])
                continue
        item["forget_reference"] = ref.rstrip(".!?")

    if invalid:
        raise ValueError(f"missing or invalid generated references for {len(invalid)} item(s): {invalid}")

    for path, data, _items in files:
        if not args.dry_run:
            dump_json(path, data)

    print(json.dumps({"updated": len(kept), "invalid": invalid}, ensure_ascii=False, indent=2))
    for item in kept:
        print(f"{item['id']}: {item['forget_reference']}")


if __name__ == "__main__":
    main()
