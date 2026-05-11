"""Re-run the LangMem battery on the new stack with progressive ask_period sweep.

For each (world, persona) we build the LangMem adapter ONCE and reuse the
underlying InMemoryStore across the three ask_periods. That works because:
  * LangMem's preload only ever appends notes — there is no reset
  * the new answer_mcq path uses store.search directly (no agent.invoke), so
    asking MCQs at one stage does not pollute the store before the next stage

Coverage matches `_legacy_buggy_langmem/` minus `no_use`:
    baseline   personas 0-3   ask_period x 3
    no_store   personas 0-3   ask_period x 3
    forget     personas 0-9   ask_period x 3

Usage:
    conda run -n langmem311 python scripts/rerun_langmem_battery.py [--workers 4] [--worlds baseline ...] [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memory_control_tests.common import (  # noqa: E402
    PERIODS,
    build_forget_stage_map,
    build_recall_summary,
    build_transformed_history_path,
    period_tag,
    rewrite_key_references,
)
from memory_control_tests.evaluation.methods import build_method_adapter  # noqa: E402
from memory_control_tests.evaluation.shared import (  # noqa: E402
    apply_world_transform,
    build_eval_prompt,
    build_label_map,
    build_persona_system_message,
    extract_choice,
    load_openai_client,
    load_sidecar,
    request_text,
)
from memory_control_tests.transforms import build_context_messages  # noqa: E402


MODEL = "gpt-5.4-mini"
MODEL_FILENAME_TAG = "gpt_5_4_mini"
RENDERED_DIR = REPO_ROOT / "data" / "test" / "travelPlanning" / "specs"
EVAL_RESULTS_ROOT = REPO_ROOT / "eval_results" / "travelPlanning"

ASK_PERIODS = [
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]

WORLD_PERSONA_COUNTS = {
    "baseline": 4,
    "no_store": 4,
    "forget": 10,
}

TARGET_INSTRUCTION_PERIODS = (
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
)


def _stem_for(persona: int) -> str:
    return f"conversation_travelPlanning_persona{persona}_sample0"


def _rendered_path(persona: int) -> Path:
    return RENDERED_DIR / f"{_stem_for(persona)}.recall_rendered.json"


def _output_path_for(world: str, persona: int, ask_period: str) -> Path:
    suffix_period = period_tag(ask_period) if ask_period != "Conversation Late Stage" else ""
    base = EVAL_RESULTS_ROOT / world / f"{MODEL}+LangMem"
    if suffix_period:
        return base / f"{_stem_for(persona)}.{world}.{suffix_period}.langmem_retrieval_eval_{MODEL_FILENAME_TAG}.json"
    return base / f"{_stem_for(persona)}.{world}.langmem_retrieval_eval_{MODEL_FILENAME_TAG}.json"


def _build_period_messages(data: Dict[str, Any], period: str) -> List[Dict[str, str]]:
    lines = data.get(period, [])
    out: List[Dict[str, str]] = []
    if not isinstance(lines, list):
        return out
    for line in lines:
        if not isinstance(line, str) or line.startswith("Side_Note"):
            continue
        if line.startswith("User:"):
            out.append({"role": "user", "content": line[len("User:"):].strip()})
        elif line.startswith("Assistant:"):
            out.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})
        elif line.strip():
            out.append({"role": "user", "content": line.strip()})
    return out


def _stage_batches_up_to(transformed: Dict[str, Any], ask_period: str) -> List[Dict[str, Any]]:
    if ask_period not in PERIODS:
        return []
    batches: List[Dict[str, Any]] = []
    for period in PERIODS[: PERIODS.index(ask_period) + 1]:
        msgs = _build_period_messages(transformed, period)
        if msgs:
            batches.append({"period": period, "messages": msgs})
    return batches


def _build_forget_eval_targets(sidecar: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    targets: Dict[str, Dict[str, str]] = {}
    keys = sidecar.get("key_turns", []) if isinstance(sidecar, dict) else []
    probes = sidecar.get("protected_probe_turns", []) if isinstance(sidecar, dict) else []
    for idx, period in enumerate(TARGET_INSTRUCTION_PERIODS):
        targets[period] = {
            "key_timestamp": str(keys[idx].get("timestamp", "")).strip() if idx < len(keys) else "",
            "probe_timestamp": str(probes[idx].get("timestamp", "")).strip() if idx < len(probes) else "",
        }
    return targets


def _should_score(world: str, ask_period: str, timestamp: str, forget_targets: Dict[str, Dict[str, str]]) -> bool:
    if world != "forget":
        return True
    stage = forget_targets.get(ask_period, {})
    allowed = {stage.get("key_timestamp", "").strip(), stage.get("probe_timestamp", "").strip()}
    allowed.discard("")
    return bool(allowed) and timestamp in allowed


def _decode(model_response: str, choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> Tuple[str, str]:
    labels = list(choices.keys())
    pc = extract_choice(model_response, labels)
    return pc, choice_to_answer_type.get(pc, "")


class _Args:
    """Minimal duck-typed args for build_method_adapter('langmem', ...)."""

    def __init__(self, *, rendered: str, world: str, model: str) -> None:
        self.rendered = rendered
        self.world = world
        self.model = model
        self.api_key_file = "keys/openrouter_key.txt"
        self.embedding_model = "text-embedding-3-small"
        self.preload_batch_size = 2
        self.memory_limit = 5


def run_persona_progressive(world: str, persona: int) -> List[Tuple[Path, bool, str]]:
    """Run all 3 ask_periods for one (world, persona), reusing one LangMem adapter."""
    rendered_path = _rendered_path(persona)
    rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    conversation_path = rendered["source_conversation"]
    conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    sidecar = load_sidecar(rendered, "")
    forget_stage_map = build_forget_stage_map(sidecar)
    forget_eval_targets = _build_forget_eval_targets(sidecar)
    label_map = build_label_map(rendered)

    rewrite_client = load_openai_client("keys/openrouter_key.txt")
    transformed_history_path = build_transformed_history_path(str(rendered_path), world)
    if transformed_history_path and transformed_history_path.exists():
        transformed = json.loads(transformed_history_path.read_text(encoding="utf-8"))
    else:
        target_refs = rewrite_key_references(
            lambda model, prompt: request_text(rewrite_client, model, [{"role": "user", "content": prompt}]),
            MODEL,
            sidecar.get("key_turns", [])[:3],
            label_map=label_map,
        )
        transformed = apply_world_transform(
            conversation, sidecar, world, target_refs,
            "Conversation Early Stage", "",
        )
        if transformed_history_path:
            transformed_history_path.parent.mkdir(parents=True, exist_ok=True)
            transformed_history_path.write_text(
                json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    persona_messages = build_persona_system_message(transformed)
    args = _Args(rendered=str(rendered_path), world=world, model=MODEL)

    adapter = build_method_adapter(
        "langmem",
        args=args,
        rendered=rendered,
        conversation_path=conversation_path,
        persona_messages=persona_messages,
        context_messages=[],
    )

    outcomes: List[Tuple[Path, bool, str]] = []
    preloaded_count = 0
    try:
        for ask_period in ASK_PERIODS:
            full_ctx = build_context_messages(transformed, ask_period)
            new_msgs = full_ctx[preloaded_count:]
            preloaded_count = len(full_ctx)
            new_stage_batches = _stage_batches_up_to(transformed, ask_period)
            # Pass only the *delta* stage batches that haven't been preloaded yet.
            already_seen_periods = {b["period"] for b in (new_stage_batches[:-1] if new_msgs else new_stage_batches)}
            del already_seen_periods  # not used; kept for clarity

            adapter.preload([], new_msgs, ask_period)

            results: Dict[str, Any] = {
                "source_rendered": str(rendered_path),
                "source_conversation": conversation_path,
                "source_sidecar": rendered.get("source_sidecar", ""),
                "method": "langmem",
                "backend": getattr(adapter, "backend_name", "langmem"),
                "model": MODEL,
                "world": world,
                "ask_period": ask_period,
                "no_use_restrict_period": "Conversation Early Stage",
                "no_use_release_period": "",
                "transformed_history_path": str(transformed_history_path) if transformed_history_path else "",
                "incremental_preload_periods": [b["period"] for b in new_stage_batches],
                "forget_eval_targets": forget_eval_targets,
                "method_debug": adapter.debug_payload(),
                "whole_recall_results": [],
                "slot_recall_results": [],
            }

            for item in rendered.get("whole_recall_set", []):
                ts = str(item.get("timestamp", "")).strip()
                if not _should_score(world, ask_period, ts, forget_eval_targets):
                    continue
                rd = item.get("rendered", {})
                choices = rd.get("choices", {})
                ctype = rd.get("choice_to_answer_type", {})
                if not choices or not ctype:
                    continue
                raw = adapter.answer_mcq(rd["question"], choices)
                resp = raw.get("model_response", "")
                pc, pt = _decode(resp, choices, ctype)
                results["whole_recall_results"].append({
                    "timestamp": item["timestamp"],
                    "turn_role": item["turn_role"],
                    "identifier_label": item["identifier_label"],
                    "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                    "question": rd["question"],
                    "choices": choices,
                    "choice_to_answer_type": ctype,
                    "model_response": resp,
                    "predicted_choice": pc,
                    "predicted_answer_type": pt,
                    "retrieved_memories": raw.get("retrieved_memories"),
                    "debug": raw.get("debug", {}),
                })

            for item in rendered.get("slot_recall_set", []):
                ts = str(item.get("timestamp", "")).strip()
                if not _should_score(world, ask_period, ts, forget_eval_targets):
                    continue
                for slot in item.get("rendered", {}).get("items", []):
                    choices = slot.get("choices", {})
                    ctype = slot.get("choice_to_answer_type", {})
                    if not choices or not ctype:
                        continue
                    raw = adapter.answer_mcq(slot["question"], choices)
                    resp = raw.get("model_response", "")
                    pc, pt = _decode(resp, choices, ctype)
                    results["slot_recall_results"].append({
                        "timestamp": item["timestamp"],
                        "turn_role": item["turn_role"],
                        "identifier_label": item["identifier_label"],
                        "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                        "sensitive_key": slot["sensitive_key"],
                        "sensitive_value": slot["sensitive_value"],
                        "question": slot["question"],
                        "choices": choices,
                        "choice_to_answer_type": ctype,
                        "model_response": resp,
                        "predicted_choice": pc,
                        "predicted_answer_type": pt,
                        "retrieved_memories": raw.get("retrieved_memories"),
                        "debug": raw.get("debug", {}),
                    })

            results["summary"] = build_recall_summary(
                world, results["whole_recall_results"], results["slot_recall_results"],
            )

            out_path = _output_path_for(world, persona, ask_period)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            outcomes.append((out_path, True, "ok"))
    finally:
        try:
            adapter.close()
        except Exception:
            pass
    return outcomes


def run_one_persona(world: str, persona: int) -> List[Tuple[Path, bool, str]]:
    try:
        return run_persona_progressive(world, persona)
    except Exception as exc:
        tb = traceback.format_exc(limit=5)
        return [(Path(f"<error:{world}/persona{persona}>"), False, f"{exc} :: {tb.splitlines()[-1]}")]


def main() -> None:
    os.chdir(str(REPO_ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", nargs="+", default=["baseline", "no_store", "forget"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_personas", type=int, default=0, help="cap personas per world (0 = no cap)")
    parser.add_argument(
        "--skip_baseline_smoked",
        action="store_true",
        help="Skip (baseline, persona0) and (baseline, persona1) — assumes their results were generated by smoke tests and copied into eval_results manually.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Build job list (one job per (world, persona); each job sweeps 3 ask_periods)
    jobs: List[Tuple[str, int]] = []
    skipped = 0
    # Hard-skip set: (world, persona) pairs we explicitly do NOT want to rerun.
    # baseline persona0 = old smoke test (agent.invoke + manage tool), persona1 = new smoke test (Option 2 dual-agent).
    # Both already have their result file copied into eval_results/.../+LangMem/baseline/.
    explicit_skip = set()
    if args.skip_baseline_smoked:
        explicit_skip.update([("baseline", 0), ("baseline", 1)])
    for world in args.worlds:
        n = WORLD_PERSONA_COUNTS.get(world, 0)
        if args.max_personas:
            n = min(n, args.max_personas)
        for p in range(n):
            if (world, p) in explicit_skip:
                skipped += 1
                continue
            # Skip if all 3 ask_period outputs already exist
            if all(_output_path_for(world, p, ap).exists() for ap in ASK_PERIODS):
                skipped += 1
                continue
            jobs.append((world, p))

    print(f"Jobs to run: {len(jobs)}  skipped (already complete): {skipped}")
    for w, p in jobs:
        print(f"  {w} persona{p}")
    if args.dry_run:
        return

    completed = 0
    total_files_ok = 0
    total_files_fail = 0
    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one_persona, w, p): (w, p) for w, p in jobs}
        for fut in as_completed(futures):
            w, p = futures[fut]
            outcomes = fut.result()
            completed += 1
            for path, ok, msg in outcomes:
                if ok:
                    total_files_ok += 1
                else:
                    total_files_fail += 1
                    failed.append(f"{w} persona{p}: {msg}")
            print(f"[{completed}/{len(jobs)}] {w} persona{p} done — {sum(1 for _,o,_ in outcomes if o)}/{len(outcomes)} files ok")

    print()
    print(f"Files: {total_files_ok} ok / {total_files_fail} fail across {completed} jobs.")
    for f in failed[:20]:
        print(f"  ! {f}")


if __name__ == "__main__":
    main()
