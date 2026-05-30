from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from ..common import (
    PERIODS,
    build_forget_stage_map,
    build_recall_summary,
    build_transformed_history_path,
)
from ..transforms import build_context_messages
from .config import EvalConfig
from .methods import build_method_adapter
from .paths import default_output_path
from .shared import (
    MEMORY_CONTROL_METADATA_KEY,
    MEMORY_CONTROL_STAGE_TRANSFORM_VERSION,
    MEMORY_CONTROL_TRANSFORM_VERSION,
    apply_world_transform,
    build_label_map,
    build_persona_system_message,
    load_sidecar,
)
from .tasks import (
    build_forget_eval_targets,
    build_incremental_stage_batches,
    build_mcq_tasks,
    decode_answer,
    run_mcq_tasks,
)


def _safe_answer_mcq(adapter: Any, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = adapter.answer_mcq(question, choices)
        elapsed = time.monotonic() - started
        debug = result.get("debug", {})
        if not isinstance(debug, dict):
            debug = {}
        debug["elapsed_seconds"] = round(elapsed, 3)
        result["debug"] = debug
        return result
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        return {
            "model_response": "",
            "retrieved_memories": None,
            "debug": {
                "elapsed_seconds": round(elapsed, 3),
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _resolve_full_conversation_path(path: str) -> str:
    source = Path(path)
    parts = source.parts
    try:
        marker_idx = next(idx for idx, part in enumerate(parts) if part.startswith("tmp_stage") and part.endswith("_source"))
    except StopIteration:
        return path
    full_path = Path(*parts[:marker_idx]) / parts[marker_idx + 1] / parts[marker_idx + 2] / source.name
    if full_path.exists():
        return str(full_path)
    return path


def _is_fresh_transformed_history(
    cached: Dict[str, Any],
    world: str,
    sidecar: Dict[str, Any],
    stage_id: str = "",
) -> bool:
    if world == "baseline":
        return True
    if world not in {"forget", "no_store", "no_use", "no_use_active", "no_use_release"}:
        return True
    metadata = cached.get(MEMORY_CONTROL_METADATA_KEY, {}) if isinstance(cached, dict) else {}
    if not isinstance(metadata, dict):
        return False
    expected_version = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION if stage_id else MEMORY_CONTROL_TRANSFORM_VERSION
    if metadata.get("transform_version") != expected_version:
        return False

    key_turns = sidecar.get("key_turns", []) if isinstance(sidecar, dict) else []
    if stage_id:
        key_turns = [
            turn for turn in key_turns
            if str((turn or {}).get("stage_id", "")).strip() == stage_id
        ]
    key_count = len(key_turns)
    if world == "forget":
        return len(metadata.get("forget_insertions", [])) == key_count
    if world == "no_store":
        return len(metadata.get("no_store_insertions", [])) == key_count
    if world in {"no_use", "no_use_active", "no_use_release"}:
        policy = metadata.get("no_use_policy", {})
        if stage_id:
            return (
                bool(metadata.get("no_use_insertions", []))
                and isinstance(policy, dict)
                and policy.get("placement") == "deterministic_random_after_key_turn_same_stage"
            )
        return bool(metadata.get("no_use_insertions", []))
    return True


def _target_references_from_labels(
    sidecar: Dict[str, Any],
    label_map: Dict[str, str],
    stage_id: str = "",
) -> List[str]:
    refs: List[str] = []
    for turn in sidecar.get("key_turns", []) if isinstance(sidecar, dict) else []:
        if stage_id and str((turn or {}).get("stage_id", "")).strip() != stage_id:
            continue
        timestamp = str((turn or {}).get("timestamp", "")).strip()
        label = str(label_map.get(timestamp, "")).strip()
        if label:
            refs.append(f"the {label[0].lower() + label[1:]}")
            continue
        refs.append(
            str(
                (turn or {}).get("key_phrase")
                or (turn or {}).get("task_goal")
                or "that earlier request"
            ).strip()
        )
    return refs


def _load_transformed_conversation(
    *,
    config: EvalConfig,
    rendered: Dict[str, Any],
    conversation: Dict[str, Any],
    sidecar: Dict[str, Any],
) -> tuple[Dict[str, Any], Path | None]:
    transformed_history_path = build_transformed_history_path(
        config.rendered,
        config.world,
        stage_id=config.stage_id,
    )
    if transformed_history_path and transformed_history_path.exists():
        cached = json.loads(transformed_history_path.read_text(encoding="utf-8"))
        if _is_fresh_transformed_history(cached, config.world, sidecar, stage_id=config.stage_id):
            return cached, transformed_history_path

    label_map = build_label_map(rendered)
    target_references = _target_references_from_labels(sidecar, label_map, stage_id=config.stage_id)
    transformed = apply_world_transform(
        conversation,
        sidecar,
        config.world,
        target_references,
        "all_stages",
        "",
        stage_id=config.stage_id,
    )
    if transformed_history_path:
        transformed_history_path.parent.mkdir(parents=True, exist_ok=True)
        transformed_history_path.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")
    return transformed, transformed_history_path


def _load_rendered(path: str) -> Dict[str, Any]:
    rendered_path = Path(path)
    rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    if rendered_path.name != "whole_recall.json":
        return rendered

    slot_path = rendered_path.with_name("slot_recall.json")
    slot_rendered: Dict[str, Any] = {}
    if slot_path.exists():
        slot_rendered = json.loads(slot_path.read_text(encoding="utf-8"))

    combined = {
        "source_rendered": rendered.get("source_rendered", str(rendered_path)),
        "source_sidecar": rendered.get("source_sidecar") or slot_rendered.get("source_sidecar", ""),
        "source_memory_targets": rendered.get("source_memory_targets")
        or slot_rendered.get("source_memory_targets", ""),
        "source_conversation": rendered.get("source_conversation")
        or slot_rendered.get("source_conversation", ""),
        "model": rendered.get("model") or slot_rendered.get("model", ""),
        "whole_recall_set": rendered.get("items", []),
        "slot_recall_set": slot_rendered.get("items", []),
    }
    return combined


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    args = config.adapter_args()
    rendered = _load_rendered(config.rendered)
    conversation_path = _resolve_full_conversation_path(rendered["source_conversation"])
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = load_sidecar(rendered, config.sidecar)
    forget_eval_targets = build_forget_eval_targets(sidecar, stage_id=config.stage_id)

    transformed_conversation, transformed_history_path = _load_transformed_conversation(
        config=config,
        rendered=rendered,
        conversation=conversation,
        sidecar=sidecar,
    )
    forget_stage_map = build_forget_stage_map(sidecar, transformed_conversation)
    persona_messages = build_persona_system_message(transformed_conversation)
    context_messages = build_context_messages(
        transformed_conversation,
        config.ask_period,
        stage_id=config.stage_id,
    )
    stage_batches = build_incremental_stage_batches(
        transformed_conversation,
        config.ask_period,
        stage_id=config.stage_id,
    )

    adapter = build_method_adapter(
        config.method,
        args=args,
        rendered=rendered,
        conversation_path=conversation_path,
        persona_messages=persona_messages,
        context_messages=context_messages,
        transformed_conversation=transformed_conversation,
    )
    adapter.preload(stage_batches, context_messages, config.ask_period)

    results = {
        "source_rendered": config.rendered,
        "source_conversation": conversation_path,
        "source_sidecar": rendered.get("source_sidecar", config.sidecar),
        "method": config.method,
        "method_config": config.method_config,
        "backend": getattr(adapter, "backend_name", config.method),
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "world": config.world,
        "ask_period": config.ask_period,
        "stage_id": config.stage_id,
        "transformed_history_path": str(transformed_history_path) if transformed_history_path else "",
        "incremental_preload_periods": [batch["period"] for batch in stage_batches],
        "forget_eval_targets": forget_eval_targets,
        "memory_control_metadata": transformed_conversation.get("_memory_control_metadata", {}),
        "method_debug": adapter.debug_payload(),
        "whole_recall_results": [],
        "slot_recall_results": [],
    }

    whole_tasks, slot_tasks = build_mcq_tasks(
        rendered=rendered,
        world=config.world,
        ask_period=config.ask_period,
        forget_targets=forget_eval_targets,
        forget_stage_map=forget_stage_map,
        stage_id=config.stage_id,
        memory_control_metadata=transformed_conversation.get("_memory_control_metadata", {}),
    )

    def run_whole(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = _safe_answer_mcq(adapter, payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = decode_answer(
            model_response, payload["choices"], payload["choice_to_answer_type"],
        )
        return {
            "timestamp": payload["timestamp"],
            "turn_role": payload["turn_role"],
            "identifier_label": payload["identifier_label"],
            "forget_stage": payload["forget_stage"],
            "evaluation_role": payload.get("evaluation_role", ""),
            "control_condition": payload.get("control_condition", ""),
            "question": payload["question"],
            "choices": payload["choices"],
            "choice_to_answer_type": payload["choice_to_answer_type"],
            "expected_choice": payload.get("expected_choice", ""),
            "expected_answer_type": payload.get("expected_answer_type", ""),
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "is_expected": predicted_answer_type == payload.get("expected_answer_type", ""),
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    def run_slot(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = _safe_answer_mcq(adapter, payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = decode_answer(
            model_response, payload["choices"], payload["choice_to_answer_type"],
        )
        return {
            "timestamp": payload["timestamp"],
            "turn_role": payload["turn_role"],
            "identifier_label": payload["identifier_label"],
            "forget_stage": payload["forget_stage"],
            "evaluation_role": payload.get("evaluation_role", ""),
            "control_condition": payload.get("control_condition", ""),
            "sensitive_key": payload["sensitive_key"],
            "sensitive_value": payload["sensitive_value"],
            "question": payload["question"],
            "choices": payload["choices"],
            "choice_to_answer_type": payload["choice_to_answer_type"],
            "expected_choice": payload.get("expected_choice", ""),
            "expected_answer_type": payload.get("expected_answer_type", ""),
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "is_expected": predicted_answer_type == payload.get("expected_answer_type", ""),
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    try:
        whole_results, slot_results = run_mcq_tasks(
            adapter=adapter,
            whole_tasks=whole_tasks,
            slot_tasks=slot_tasks,
            run_whole=run_whole,
            run_slot=run_slot,
            workers=config.workers,
        )
        results["whole_recall_results"] = whole_results
        results["slot_recall_results"] = slot_results
    finally:
        adapter.close()

    results["summary"] = build_recall_summary(
        config.world,
        results["whole_recall_results"],
        results["slot_recall_results"],
    )
    return results


def write_evaluation_results(config: EvalConfig, results: Dict[str, Any]) -> str:
    output_path = config.output or default_output_path(
        config.rendered,
        config.world,
        config.ask_period,
        config.method,
        config.model,
        stage_id=config.stage_id,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


# ----------------------------------------------------------------------
# Multi-stage incremental evaluation.
#
# Restores the cross-period qdrant reuse that the OLD per-method runners had
# (and that CLAUDE.md's "Memory is incremental across stages" convention
# expects). One process per (persona, world): preload Initial+Early -> answer
# Early MCQs -> add Intermediate -> answer Intermediate MCQs -> add Late ->
# answer Late MCQs. Each period writes its own per-stage output file in the
# same shape `run_evaluation` produces, so downstream report aggregation does
# not need to change.
# ----------------------------------------------------------------------


def _run_mcqs_at_period(
    *,
    config: EvalConfig,
    ask_period: str,
    rendered: Dict[str, Any],
    conversation_path: str,
    transformed_history_path: Path | None,
    transformed_conversation: Dict[str, Any],
    forget_stage_map: Dict[str, Any],
    forget_eval_targets: Dict[str, Any],
    adapter: Any,
    incremental_preload_periods: List[str],
) -> Dict[str, Any]:
    """Build MCQ tasks for one ask_period, run them through `adapter`, build
    the same results dict shape `run_evaluation` produces, and return it.

    Caller is responsible for the preload state (the adapter must already have
    ingested the conversation up to `ask_period` before calling).
    """
    results = {
        "source_rendered": config.rendered,
        "source_conversation": conversation_path,
        "source_sidecar": rendered.get("source_sidecar", config.sidecar),
        "method": config.method,
        "method_config": config.method_config,
        "backend": getattr(adapter, "backend_name", config.method),
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "world": config.world,
        "ask_period": ask_period,
        "transformed_history_path": str(transformed_history_path) if transformed_history_path else "",
        "incremental_preload_periods": list(incremental_preload_periods),
        "forget_eval_targets": forget_eval_targets,
        "memory_control_metadata": transformed_conversation.get("_memory_control_metadata", {}),
        "method_debug": copy.deepcopy(adapter.debug_payload()),
        "whole_recall_results": [],
        "slot_recall_results": [],
    }

    whole_tasks, slot_tasks = build_mcq_tasks(
        rendered=rendered,
        world=config.world,
        ask_period=ask_period,
        forget_targets=forget_eval_targets,
        forget_stage_map=forget_stage_map,
    )

    def run_whole(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = _safe_answer_mcq(adapter, payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = decode_answer(
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
            "expected_choice": payload.get("expected_choice", ""),
            "expected_answer_type": payload.get("expected_answer_type", ""),
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "is_expected": predicted_answer_type == payload.get("expected_answer_type", ""),
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    def run_slot(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = _safe_answer_mcq(adapter, payload["question"], payload["choices"])
        model_response = raw_result.get("model_response", "")
        predicted_choice, predicted_answer_type = decode_answer(
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
            "expected_choice": payload.get("expected_choice", ""),
            "expected_answer_type": payload.get("expected_answer_type", ""),
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "is_expected": predicted_answer_type == payload.get("expected_answer_type", ""),
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    whole_results, slot_results = run_mcq_tasks(
        adapter=adapter,
        whole_tasks=whole_tasks,
        slot_tasks=slot_tasks,
        run_whole=run_whole,
        run_slot=run_slot,
        workers=config.workers,
    )
    results["whole_recall_results"] = whole_results
    results["slot_recall_results"] = slot_results

    results["summary"] = build_recall_summary(
        config.world,
        results["whole_recall_results"],
        results["slot_recall_results"],
    )
    return results


def run_multi_stage_evaluation(config: EvalConfig) -> List[Dict[str, Any]]:
    """Drive a single (persona, world) over multiple ask_periods incrementally.

    Builds the adapter ONCE, then for each period in `config.ask_periods` (in
    canonical order Initial -> Early -> Intermediate -> Late):
      * adds only the NEW stage batches since the previous period
      * snapshots the adapter
      * runs MCQ tasks for that ask_period
      * writes the per-period output file

    Returns the list of (period, written-path, results-dict) tuples for the
    caller to log.
    """
    if not config.ask_periods:
        raise ValueError("run_multi_stage_evaluation requires config.ask_periods to be non-empty")

    # Order by PERIODS so callers can pass them in any order.
    ordered = [p for p in PERIODS if p in config.ask_periods]
    if not ordered:
        raise ValueError(f"no valid periods in {config.ask_periods}")
    final_period = ordered[-1]

    # Build everything against the FINAL period so the conversation transform
    # / persona / context covers the full requested span.
    base_args = config.adapter_args()
    base_args.ask_period = final_period
    rendered = json.loads(Path(config.rendered).read_text(encoding="utf-8"))
    conversation_path = _resolve_full_conversation_path(rendered["source_conversation"])
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = load_sidecar(rendered, config.sidecar)
    forget_eval_targets = build_forget_eval_targets(sidecar)

    final_config = replace(config, ask_period=final_period)
    transformed_conversation, transformed_history_path = _load_transformed_conversation(
        config=final_config,
        rendered=rendered,
        conversation=conversation,
        sidecar=sidecar,
    )
    forget_stage_map = build_forget_stage_map(sidecar, transformed_conversation)
    persona_messages = build_persona_system_message(transformed_conversation)
    full_context_messages = build_context_messages(transformed_conversation, final_period)
    full_stage_batches = build_incremental_stage_batches(transformed_conversation, final_period)

    adapter = build_method_adapter(
        config.method,
        args=base_args,
        rendered=rendered,
        conversation_path=conversation_path,
        persona_messages=persona_messages,
        context_messages=full_context_messages,
        transformed_conversation=transformed_conversation,
    )

    written: List[Dict[str, Any]] = []
    try:
        # Index stage batches by period for incremental slicing.
        batch_by_period = {b["period"]: b for b in full_stage_batches}
        # Periods we need to ensure are loaded INTO the store before each
        # eval. Always start by walking from the beginning of PERIODS.
        already_loaded: List[str] = []
        for ap in ordered:
            cur_idx = PERIODS.index(ap)
            # Stages strictly between (last_loaded, ap]: feed them now.
            target_stages = PERIODS[: cur_idx + 1]
            new_periods = [p for p in target_stages if p not in already_loaded]
            new_batches = [batch_by_period[p] for p in new_periods if p in batch_by_period]
            if new_batches:
                ctx_at_ap = build_context_messages(transformed_conversation, ap)
                adapter.preload(new_batches, ctx_at_ap, ap)
                already_loaded.extend(new_periods)

            ap_config = replace(config, ask_period=ap)
            results = _run_mcqs_at_period(
                config=ap_config,
                ask_period=ap,
                rendered=rendered,
                conversation_path=conversation_path,
                transformed_history_path=transformed_history_path,
                transformed_conversation=transformed_conversation,
                forget_stage_map=forget_stage_map,
                forget_eval_targets=forget_eval_targets,
                adapter=adapter,
                incremental_preload_periods=list(already_loaded),
            )
            out_path = write_evaluation_results(ap_config, results)
            written.append({"ask_period": ap, "output": out_path, "results": results})
    finally:
        adapter.close()

    return written
