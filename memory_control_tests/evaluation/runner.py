from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..common import (
    build_forget_stage_map,
    build_recall_summary,
    build_transformed_history_path,
    rewrite_key_references,
)
from ..transforms import build_context_messages
from .config import EvalConfig
from .methods import build_method_adapter
from .paths import default_output_path
from .shared import (
    apply_world_transform,
    build_label_map,
    build_persona_system_message,
    load_openai_client,
    load_sidecar,
    request_text,
)
from .tasks import (
    build_forget_eval_targets,
    build_incremental_stage_batches,
    build_mcq_tasks,
    decode_answer,
    run_mcq_tasks,
)


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
        release_period=config.no_use_release_period or None,
        restrict_period=config.no_use_restrict_period,
    )
    if transformed_history_path and transformed_history_path.exists():
        return json.loads(transformed_history_path.read_text(encoding="utf-8")), transformed_history_path

    label_map = build_label_map(rendered)
    rewrite_client = load_openai_client(config.api_key_file)
    target_references = rewrite_key_references(
        lambda model, prompt: request_text(rewrite_client, model, [{"role": "user", "content": prompt}]),
        config.model,
        sidecar.get("key_turns", [])[:3],
        label_map=label_map,
    )
    transformed = apply_world_transform(
        conversation,
        sidecar,
        config.world,
        target_references,
        config.no_use_restrict_period,
        config.no_use_release_period,
    )
    if transformed_history_path:
        transformed_history_path.parent.mkdir(parents=True, exist_ok=True)
        transformed_history_path.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")
    return transformed, transformed_history_path


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    args = config.adapter_args()
    rendered = json.loads(Path(config.rendered).read_text(encoding="utf-8"))
    conversation_path = rendered["source_conversation"]
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = load_sidecar(rendered, config.sidecar)
    forget_stage_map = build_forget_stage_map(sidecar)
    forget_eval_targets = build_forget_eval_targets(sidecar)

    transformed_conversation, transformed_history_path = _load_transformed_conversation(
        config=config,
        rendered=rendered,
        conversation=conversation,
        sidecar=sidecar,
    )
    persona_messages = build_persona_system_message(transformed_conversation)
    context_messages = build_context_messages(transformed_conversation, config.ask_period)
    stage_batches = build_incremental_stage_batches(transformed_conversation, config.ask_period)

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
        "no_use_restrict_period": config.no_use_restrict_period,
        "no_use_release_period": config.no_use_release_period,
        "transformed_history_path": str(transformed_history_path) if transformed_history_path else "",
        "incremental_preload_periods": [batch["period"] for batch in stage_batches],
        "forget_eval_targets": forget_eval_targets,
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
    )

    def run_whole(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = adapter.answer_mcq(payload["question"], payload["choices"])
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
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
            "retrieved_memories": raw_result.get("retrieved_memories"),
            "debug": raw_result.get("debug", {}),
        }

    def run_slot(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = adapter.answer_mcq(payload["question"], payload["choices"])
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
            "model_response": model_response,
            "predicted_choice": predicted_choice,
            "predicted_answer_type": predicted_answer_type,
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
        no_use_restrict_period=config.no_use_restrict_period,
        no_use_release_period=config.no_use_release_period,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
