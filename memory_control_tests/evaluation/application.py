from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cost import log_cost_delta, snapshot_openrouter_usage
from .shared import (
    extract_choice,
    load_openai_client,
    mark_cache_breakpoint,
    request_text,
    resolve_model_name,
)


DEFAULT_DATA_ROOT = Path("data/application/mcq/by_world")
DEFAULT_OUTPUT_ROOT = Path("eval_results/application")


def _default_output_path(input_path: Path, model: str) -> Path:
    model_tag = str(model).replace("/", "_")
    world = input_path.stem
    return DEFAULT_OUTPUT_ROOT / world / model_tag / f"{world}.plain_api_{model_tag}.json"


def _load_world(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Application world file must contain a JSON object: {path}")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Application world file is missing list field 'items': {path}")
    return data


def _messages_for_request(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Application item is missing non-empty messages: {item.get('id', '<unknown>')}")
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid message in item {item.get('id', '<unknown>')}: {message!r}")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Invalid message shape in item {item.get('id', '<unknown>')}: {message!r}")
        normalized.append({"role": role, "content": content})
    if len(normalized) == 1:
        return normalized
    return mark_cache_breakpoint(normalized[:-1]) + [normalized[-1]]


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(results),
        "correct": sum(1 for row in results if row.get("is_expected")),
        "parse_failures": sum(1 for row in results if not row.get("predicted_choice")),
        "by_topic": {},
    }
    summary["accuracy"] = summary["correct"] / summary["total"] if summary["total"] else 0.0
    by_topic: Dict[str, Dict[str, Any]] = summary["by_topic"]
    for row in results:
        topic = str(row.get("topic") or "unknown")
        bucket = by_topic.setdefault(topic, {"total": 0, "correct": 0, "parse_failures": 0})
        bucket["total"] += 1
        if row.get("is_expected"):
            bucket["correct"] += 1
        if not row.get("predicted_choice"):
            bucket["parse_failures"] += 1
    for bucket in by_topic.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    return summary


def _run_one_item(
    *,
    client: Any,
    model: str,
    item: Dict[str, Any],
    reasoning_effort: str,
    request_timeout: Optional[int],
) -> Dict[str, Any]:
    started = time.time()
    choices = item.get("choices") if isinstance(item.get("choices"), dict) else {}
    labels = [str(label).upper() for label in choices.keys()]
    expected_choice = str(item.get("expected_choice", "")).upper()
    last_exc: Optional[Exception] = None
    response = ""
    for attempt in range(4):
        try:
            response = request_text(
                client,
                model,
                _messages_for_request(item),
                reasoning_effort=reasoning_effort or None,
                timeout=request_timeout,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        if last_exc is not None:
            raise last_exc
    predicted_choice = extract_choice(response, labels)
    return {
        "id": item.get("id"),
        "topic": item.get("topic"),
        "stage_id": item.get("stage_id"),
        "target_turn_id": item.get("target_turn_id"),
        "world": item.get("world"),
        "expected_choice": expected_choice,
        "predicted_choice": predicted_choice,
        "is_expected": bool(predicted_choice and predicted_choice == expected_choice),
        "expected_behavior": item.get("expected_behavior"),
        "choice_roles": item.get("choice_roles"),
        "question": item.get("question"),
        "choices": choices,
        "model_response": response,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def run_application_eval(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    api_key_file: str,
    workers: int,
    request_timeout: int,
    reasoning_effort: str = "",
    overwrite: bool = False,
) -> Dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    data = _load_world(input_path)
    items = data["items"]
    client = load_openai_client(api_key_file)
    timeout_value = request_timeout if request_timeout and request_timeout > 0 else None
    max_workers = max(1, workers)

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _run_one_item,
                client=client,
                model=model,
                item=item,
                reasoning_effort=reasoning_effort,
                request_timeout=timeout_value,
            ): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                item = items[idx]
                results[idx] = {
                    "id": item.get("id"),
                    "topic": item.get("topic"),
                    "stage_id": item.get("stage_id"),
                    "target_turn_id": item.get("target_turn_id"),
                    "world": item.get("world"),
                    "expected_choice": str(item.get("expected_choice", "")).upper(),
                    "predicted_choice": "",
                    "is_expected": False,
                    "expected_behavior": item.get("expected_behavior"),
                    "choice_roles": item.get("choice_roles"),
                    "question": item.get("question"),
                    "choices": item.get("choices") if isinstance(item.get("choices"), dict) else {},
                    "model_response": f"<error: {exc}>",
                    "elapsed_seconds": 0.0,
                }

    complete_results = [row for row in results if row is not None]
    payload = {
        "dataset": "application",
        "input": str(input_path),
        "world": data.get("world") or input_path.stem,
        "method": "plain_api",
        "model": model,
        "resolved_model": resolve_model_name(model),
        "workers": max_workers,
        "request_timeout": request_timeout,
        "summary": _summarize_results(complete_results),
        "results": complete_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plain API evaluation for application MCQ world files.")
    parser.add_argument("--input", required=True, help="Path to data/application/mcq/by_world/<world>.json")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults under eval_results/application/.")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api_key_file", default="keys/openrouter_key.txt")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--reasoning_effort", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path, args.model)
    cost_start = snapshot_openrouter_usage(args.api_key_file)
    try:
        payload = run_application_eval(
            input_path=input_path,
            output_path=output_path,
            model=args.model,
            api_key_file=args.api_key_file,
            workers=args.workers,
            request_timeout=args.request_timeout,
            reasoning_effort=args.reasoning_effort,
            overwrite=args.overwrite,
        )
        summary = payload["summary"]
        print(
            f"{output_path} total={summary['total']} correct={summary['correct']} "
            f"accuracy={summary['accuracy']:.3f} parse_failures={summary['parse_failures']}"
        )
    finally:
        log_cost_delta(cost_start, snapshot_openrouter_usage(args.api_key_file))


if __name__ == "__main__":
    main()
