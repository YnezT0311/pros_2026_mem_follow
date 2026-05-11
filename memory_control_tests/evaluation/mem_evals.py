import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ..common import (
    PERIODS,
    TARGET_INSTRUCTION_PERIODS,
    build_forget_stage_map,
    build_recall_summary,
    build_transformed_history_path,
    period_tag,
    rewrite_key_references,
)
from ..transforms import build_context_messages
from .methods import build_method_adapter
from .shared import (
    apply_world_transform,
    build_label_map,
    build_persona_system_message,
    extract_choice,
    load_openai_client,
    load_sidecar,
    request_text,
)


def _build_period_messages(data: Dict[str, Any], period: str) -> List[Dict[str, str]]:
    lines = data.get(period, [])
    messages: List[Dict[str, str]] = []
    if not isinstance(lines, list):
        return messages
    for line in lines:
        if not isinstance(line, str) or line.startswith("Side_Note"):
            continue
        if line.startswith("User:"):
            messages.append({"role": "user", "content": line[len("User:"):].strip()})
        elif line.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})
        elif line.strip():
            messages.append({"role": "user", "content": line.strip()})
    return messages


def _build_incremental_stage_batches(data: Dict[str, Any], ask_period: str) -> List[Dict[str, Any]]:
    if ask_period not in PERIODS:
        return []
    batches: List[Dict[str, Any]] = []
    for period in PERIODS[: PERIODS.index(ask_period) + 1]:
        period_messages = _build_period_messages(data, period)
        if period_messages:
            batches.append({"period": period, "messages": period_messages})
    return batches


def _build_forget_eval_targets(sidecar: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    targets: Dict[str, Dict[str, str]] = {}
    key_turns = sidecar.get("key_turns", []) if isinstance(sidecar, dict) else []
    probe_turns = sidecar.get("protected_probe_turns", []) if isinstance(sidecar, dict) else []
    for idx, period in enumerate(TARGET_INSTRUCTION_PERIODS):
        key_timestamp = str(key_turns[idx].get("timestamp", "")).strip() if idx < len(key_turns) else ""
        probe_timestamp = str(probe_turns[idx].get("timestamp", "")).strip() if idx < len(probe_turns) else ""
        targets[period] = {
            "key_timestamp": key_timestamp,
            "probe_timestamp": probe_timestamp,
        }
    return targets


def _should_score_item_for_world(
    *,
    world: str,
    ask_period: str,
    timestamp: str,
    forget_targets: Dict[str, Dict[str, str]],
) -> bool:
    if world != "forget":
        return True
    stage_targets = forget_targets.get(ask_period, {})
    allowed_timestamps = {
        str(stage_targets.get("key_timestamp", "")).strip(),
        str(stage_targets.get("probe_timestamp", "")).strip(),
    }
    allowed_timestamps.discard("")
    if not allowed_timestamps:
        return False
    return timestamp in allowed_timestamps


def _run_mcq_tasks(
    *,
    adapter: Any,
    whole_tasks: List[Tuple[int, Dict[str, Any]]],
    slot_tasks: List[Tuple[Tuple[int, int], Dict[str, Any]]],
    run_whole: Callable[[Dict[str, Any]], Dict[str, Any]],
    run_slot: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Dispatch MCQ work either sequentially or via a thread pool.

    Memory-backed adapters share state across calls, so they must run
    sequentially. Plain (and any future stateless) adapters declare
    `supports_parallel_mcq = True` to opt in to the thread-pool path.
    """
    parallel = bool(getattr(adapter, "supports_parallel_mcq", False)) and workers > 1
    if not parallel:
        whole_results = [run_whole(payload) for _, payload in whole_tasks]
        slot_results = [run_slot(payload) for _, payload in slot_tasks]
        return whole_results, slot_results

    whole_results_by_idx: Dict[int, Dict[str, Any]] = {}
    slot_results_by_idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        whole_futures = {executor.submit(run_whole, payload): idx for idx, payload in whole_tasks}
        slot_futures = {executor.submit(run_slot, payload): idx for idx, payload in slot_tasks}
        for future in as_completed(list(whole_futures.keys()) + list(slot_futures.keys())):
            if future in whole_futures:
                whole_results_by_idx[whole_futures[future]] = future.result()
            else:
                slot_results_by_idx[slot_futures[future]] = future.result()
    whole_results = [whole_results_by_idx[idx] for idx, _ in whole_tasks]
    slot_results = [slot_results_by_idx[idx] for idx, _ in slot_tasks]
    return whole_results, slot_results


METHOD_FILENAME_TAG = {
    "plain": "raw_eval",
    "mem0": "mem0_retrieval_eval",
    "amem": "a_mem_retrieval_eval",
    "langmem": "langmem_retrieval_eval",
    "zep": "zep_retrieval_eval",
    "memoryos": "memoryos_retrieval_eval",
    "memtree": "memtree_retrieval_eval",
}

METHOD_ADAPTER_DIR_TAG = {
    "plain": "",
    "mem0": "mem0",
    "amem": "A-Mem",
    "langmem": "LangMem",
    "zep": "Zep",
    "memoryos": "MemoryOS",
    "memtree": "MemTree",
}


def _model_tag_for_filename(method: str, model: str) -> str:
    if method == "langmem":
        return "".join(ch if ch.isalnum() else "_" for ch in str(model))
    return str(model).replace("/", "_")


def _topic_from_rendered(rendered: str) -> str:
    # rendered convention: data/test/<topic>/specs/<stem>.recall_rendered.json
    parts = Path(rendered).parts
    if len(parts) >= 3 and parts[-2] == "specs":
        return parts[-3]
    raise ValueError(
        f"Cannot infer topic from rendered path {rendered!r}; expected "
        "'<...>/data/test/<topic>/specs/<stem>.recall_rendered.json'."
    )


def _default_output_path(
    rendered: str,
    world: str,
    ask_period: str,
    method: str,
    model: str,
    *,
    no_use_restrict_period: str = "",
    no_use_release_period: str = "",
) -> str:
    """Default output path under eval_results/<topic>/<world>/<model>[+Adapter]/.

    Filenames embed `<method>_eval_<model>` so `summarize_instruction_control_results.py`
    can identify the backend by name (see `_backend_from_name`).
    """
    method_tag = METHOD_FILENAME_TAG.get(method, f"{method}_eval")
    model_tag = _model_tag_for_filename(method, model)
    if world == "no_use":
        suffix = f".{world}.restrict_{period_tag(no_use_restrict_period or 'Conversation Early Stage')}"
        if no_use_release_period:
            suffix += f".release_{period_tag(no_use_release_period)}"
        suffix += f".test_{period_tag(ask_period)}.{method_tag}_{model_tag}.json"
    else:
        suffix = f".{world}.{method_tag}_{model_tag}.json"
        if ask_period != "Conversation Late Stage":
            suffix = f".{world}.{period_tag(ask_period)}.{method_tag}_{model_tag}.json"

    topic = _topic_from_rendered(rendered)
    stem = Path(rendered).name[: -len(".recall_rendered.json")]
    adapter_tag = METHOD_ADAPTER_DIR_TAG.get(method, method)
    model_dir = f"{str(model).replace('/', '_')}"
    folder = f"{model_dir}+{adapter_tag}" if adapter_tag else model_dir
    return str(Path("eval_results") / topic / world / folder / f"{stem}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory-system evaluations and collect raw responses/debug output.")
    parser.add_argument("--rendered", required=True)
    parser.add_argument("--method", choices=["plain", "mem0", "langmem", "amem", "zep", "memoryos", "memtree"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--ask_period", default="Conversation Late Stage")
    parser.add_argument("--world", choices=["baseline", "no_store", "forget", "no_use"], default="baseline")
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--api_key_file", default="keys/openrouter_key.txt")
    parser.add_argument(
        "--reasoning_effort",
        default="",
        help="Forward `reasoning: {effort: ...}` to OpenRouter (low|medium|high). "
             "Empty = leave provider default (off for Anthropic/OpenAI, on for Gemini).",
    )
    parser.add_argument("--memory_limit", type=int, default=5)
    parser.add_argument("--no_use_restrict_period", default="Conversation Early Stage")
    parser.add_argument("--no_use_release_period", default="")
    parser.add_argument("--embedding_model", default="")
    parser.add_argument("--preload_batch_size", type=int, default=2)
    parser.add_argument("--mem0_runtime_root", default="")
    parser.add_argument(
        "--mem0_keep_runtime",
        action="store_true",
        help="Reuse the existing self-hosted mem0 store on disk instead of resetting it.",
    )
    parser.add_argument("--zep_api_key_file", default="keys/zep_api_key.txt")
    parser.add_argument("--zep_api_url_file", default="keys/zep_api_url.txt")
    parser.add_argument("--memoryos_runtime_root", default="")
    parser.add_argument(
        "--memoryos_keep_runtime",
        action="store_true",
        help="Reuse the existing MemoryOS store on disk instead of resetting it.",
    )
    parser.add_argument("--memoryos_short_term_capacity", type=int, default=10)
    parser.add_argument("--memoryos_mid_term_capacity", type=int, default=2000)
    parser.add_argument("--memoryos_long_term_knowledge_capacity", type=int, default=100)
    parser.add_argument("--memoryos_retrieval_queue_capacity", type=int, default=7)
    parser.add_argument("--memtree_runtime_root", default="")
    parser.add_argument(
        "--memtree_keep_runtime",
        action="store_true",
        help="Reuse the existing MemTree (Milvus Lite) store on disk instead of resetting it.",
    )
    parser.add_argument("--memtree_base_threshold", type=float, default=0.4)
    parser.add_argument("--memtree_rate", type=float, default=0.5)
    parser.add_argument("--memtree_max_depth", type=int, default=15)
    parser.add_argument("--memtree_top_k_retrieve", type=int, default=10)
    parser.add_argument("--memtree_embedding_batch_size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max parallel MCQ workers. Only used when the adapter declares supports_parallel_mcq=True.",
    )
    args = parser.parse_args()
    args.mem0_reset_runtime = not args.mem0_keep_runtime
    args.memoryos_reset_runtime = not args.memoryos_keep_runtime
    args.memtree_reset_runtime = not args.memtree_keep_runtime

    if not args.embedding_model:
        if args.method == "langmem":
            args.embedding_model = "text-embedding-3-small"
        elif args.method in ("amem", "memoryos", "memtree"):
            args.embedding_model = "all-MiniLM-L6-v2"

    rendered = json.loads(Path(args.rendered).read_text(encoding="utf-8"))
    conversation_path = rendered["source_conversation"]
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = load_sidecar(rendered, args.sidecar)
    forget_stage_map = build_forget_stage_map(sidecar)
    forget_eval_targets = _build_forget_eval_targets(sidecar)
    label_map = build_label_map(rendered)
    rewrite_client = load_openai_client(args.api_key_file)
    transformed_history_path = build_transformed_history_path(
        args.rendered,
        args.world,
        release_period=args.no_use_release_period or None,
        restrict_period=args.no_use_restrict_period,
    )
    if transformed_history_path and transformed_history_path.exists():
        transformed_conversation = json.loads(transformed_history_path.read_text(encoding="utf-8"))
    else:
        target_references = rewrite_key_references(
            lambda model, prompt: request_text(rewrite_client, model, [{"role": "user", "content": prompt}]),
            args.model,
            sidecar.get("key_turns", [])[:3],
            label_map=label_map,
        )
        transformed_conversation = apply_world_transform(
            conversation,
            sidecar,
            args.world,
            target_references,
            args.no_use_restrict_period,
            args.no_use_release_period,
        )
        if transformed_history_path:
            transformed_history_path.parent.mkdir(parents=True, exist_ok=True)
            transformed_history_path.write_text(
                json.dumps(transformed_conversation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    persona_messages = build_persona_system_message(transformed_conversation)
    context_messages = build_context_messages(transformed_conversation, args.ask_period)
    stage_batches = _build_incremental_stage_batches(transformed_conversation, args.ask_period)

    adapter = build_method_adapter(
        args.method,
        args=args,
        rendered=rendered,
        conversation_path=conversation_path,
        persona_messages=persona_messages,
        context_messages=context_messages,
        transformed_conversation=transformed_conversation,
    )
    adapter.preload(stage_batches, context_messages, args.ask_period)

    results = {
        "source_rendered": args.rendered,
        "source_conversation": conversation_path,
        "source_sidecar": rendered.get("source_sidecar", args.sidecar),
        "method": args.method,
        "backend": getattr(adapter, "backend_name", args.method),
        "model": args.model,
        "reasoning_effort": getattr(args, "reasoning_effort", "") or "",
        "world": args.world,
        "ask_period": args.ask_period,
        "no_use_restrict_period": args.no_use_restrict_period,
        "no_use_release_period": args.no_use_release_period,
        "transformed_history_path": str(transformed_history_path) if transformed_history_path else "",
        "incremental_preload_periods": [batch["period"] for batch in stage_batches],
        "forget_eval_targets": forget_eval_targets,
        "method_debug": adapter.debug_payload(),
        "whole_recall_results": [],
        "slot_recall_results": [],
    }

    whole_tasks: List[Tuple[int, Dict[str, Any]]] = []
    for whole_idx, item in enumerate(rendered.get("whole_recall_set", [])):
        timestamp = str(item.get("timestamp", "")).strip()
        if not _should_score_item_for_world(
            world=args.world,
            ask_period=args.ask_period,
            timestamp=timestamp,
            forget_targets=forget_eval_targets,
        ):
            continue
        rendered_item = item.get("rendered", {})
        choices = rendered_item.get("choices", {})
        choice_to_answer_type = rendered_item.get("choice_to_answer_type", {})
        if not choices or not choice_to_answer_type:
            continue
        whole_tasks.append(
            (
                whole_idx,
                {
                    "timestamp": item["timestamp"],
                    "turn_role": item["turn_role"],
                    "identifier_label": item["identifier_label"],
                    "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                    "question": rendered_item["question"],
                    "choices": choices,
                    "choice_to_answer_type": choice_to_answer_type,
                },
            )
        )

    slot_tasks: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
    for slot_idx, item in enumerate(rendered.get("slot_recall_set", [])):
        timestamp = str(item.get("timestamp", "")).strip()
        if not _should_score_item_for_world(
            world=args.world,
            ask_period=args.ask_period,
            timestamp=timestamp,
            forget_targets=forget_eval_targets,
        ):
            continue
        for sub_idx, slot_item in enumerate(item.get("rendered", {}).get("items", [])):
            choices = slot_item.get("choices", {})
            choice_to_answer_type = slot_item.get("choice_to_answer_type", {})
            if not choices or not choice_to_answer_type:
                continue
            slot_tasks.append(
                (
                    (slot_idx, sub_idx),
                    {
                        "timestamp": item["timestamp"],
                        "turn_role": item["turn_role"],
                        "identifier_label": item["identifier_label"],
                        "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                        "sensitive_key": slot_item["sensitive_key"],
                        "sensitive_value": slot_item["sensitive_value"],
                        "question": slot_item["question"],
                        "choices": choices,
                        "choice_to_answer_type": choice_to_answer_type,
                    },
                )
            )

    def _decode(model_response: str, choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> Tuple[str, str]:
        labels = list(choices.keys())
        predicted_choice = extract_choice(model_response, labels)
        predicted_answer_type = choice_to_answer_type.get(predicted_choice, "")
        return predicted_choice, predicted_answer_type

    def run_whole(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = adapter.answer_mcq(payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = _decode(
            model_response, payload["choices"], payload["choice_to_answer_type"],
        )
        return {
            "timestamp": payload["timestamp"],
            "turn_role": payload["turn_role"],
            "identifier_label": payload["identifier_label"],
            "forget_stage": payload["forget_stage"],
            "question": payload["question"],
            "choices": payload["choices"],
            "choice_to_answer_type": payload["choice_to_answer_type"],
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    def run_slot(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = adapter.answer_mcq(payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = _decode(
            model_response, payload["choices"], payload["choice_to_answer_type"],
        )
        return {
            "timestamp": payload["timestamp"],
            "turn_role": payload["turn_role"],
            "identifier_label": payload["identifier_label"],
            "forget_stage": payload["forget_stage"],
            "sensitive_key": payload["sensitive_key"],
            "sensitive_value": payload["sensitive_value"],
            "question": payload["question"],
            "choices": payload["choices"],
            "choice_to_answer_type": payload["choice_to_answer_type"],
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    try:
        whole_results, slot_results = _run_mcq_tasks(
            adapter=adapter,
            whole_tasks=whole_tasks,
            slot_tasks=slot_tasks,
            run_whole=run_whole,
            run_slot=run_slot,
            workers=args.workers,
        )
        results["whole_recall_results"] = whole_results
        results["slot_recall_results"] = slot_results
    finally:
        adapter.close()

    results["summary"] = build_recall_summary(
        args.world,
        results["whole_recall_results"],
        results["slot_recall_results"],
    )

    output_path = args.output or _default_output_path(
        args.rendered,
        args.world,
        args.ask_period,
        args.method,
        args.model,
        no_use_restrict_period=args.no_use_restrict_period,
        no_use_release_period=args.no_use_release_period,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
