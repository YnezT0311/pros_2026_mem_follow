#!/usr/bin/env python3
"""Build per-stage application transformed worlds.

Application MCQs are evaluated on original per-stage contexts rather than a
concatenated full conversation. Each kept item expands into six worlds:

- never_seen_baseline: target block removed, expect without-memory answer
- seen_baseline: target block present, expect use-memory answer
- no_store: no-store instruction on the target turn, expect without-memory
- forget: forget instruction after the target block, expect without-memory
- no_use_active: no-use instruction active, expect without-memory
- no_use_release: no-use instruction released, expect use-memory

The rendered world records include the same message shape used by the plain
evaluation adapter: persona system message + source stage context messages +
build_eval_prompt(question, choices).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_control_tests.transforms import (
    MEMORY_CONTROL_METADATA_KEY,
    apply_no_store,
    apply_stage_local_forget,
    apply_stage_local_no_use,
)

WORLD_ORDER = [
    "never_seen_baseline",
    "seen_baseline",
    "no_store",
    "forget",
    "no_use_active",
    "no_use_release",
]

TEMPLATES = json.loads(
    (REPO_ROOT / "memory_control_tests" / "templates.json").read_text(encoding="utf-8")
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def stable_index(*parts: str) -> int:
    joined = "\n".join(str(part or "") for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def pick_template(pool: list[str], *parts: str) -> str:
    if not pool:
        return ""
    return pool[stable_index(*parts) % len(pool)]


def fill_template(text: str, *, target_reference: str) -> str:
    return text.format(target_reference=target_reference).strip()


def build_eval_prompt(question: str, choices: dict[str, str]) -> str:
    labels = list(choices.keys())
    rendered_options = [f"({label.lower()}) {choices[label]}" for label in labels]
    label_str = "/".join(label.lower() for label in labels)
    instructions = (
        "Find the most appropriate response and briefly explain your reasoning, "
        "quoting the exact words from the conversation when relevant.\n\n"
        "Output format:\n"
        "Rationale: <brief rationale citing exact conversation words when relevant>\n"
        f"<final_answer>({label_str})</final_answer>"
    )
    return f"Question: {question}\n\n" + "\n".join(rendered_options) + f"\n\n{instructions}"


def persona_messages(conversation_package: dict[str, Any]) -> list[dict[str, str]]:
    persona = str(conversation_package.get("Expanded Persona") or "").strip()
    if not persona:
        return []
    return [{"role": "system", "content": f"Current user persona: {persona}"}]


def append_sentence(text: str, sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return text
    if text.rstrip().endswith((".", "!", "?")):
        return f"{text.rstrip()} {sentence}"
    return f"{text.rstrip()}. {sentence}"


def merge_ack(text: str, ack: str) -> str:
    if not ack:
        return text
    return f"{ack} {text}".strip()


def source_stage_key(stage_id: str) -> str:
    if not stage_id.startswith("stage_"):
        raise ValueError(f"unexpected stage_id: {stage_id}")
    return f"Conversation Stage {stage_id.removeprefix('stage_')}"


def parse_source_stage(conversation_package: dict[str, Any], stage_id: str) -> list[dict[str, str]]:
    stage_key = source_stage_key(stage_id)
    raw_turns = conversation_package.get(stage_key)
    if not isinstance(raw_turns, list):
        raise ValueError(f"{stage_key} is missing or is not a list")

    messages: list[dict[str, str]] = []
    for turn in raw_turns:
        if isinstance(turn, dict):
            role = str(turn.get("role") or "").strip().lower()
            content = str(turn.get("content") or "").strip()
        else:
            text = str(turn).strip()
            if text.startswith("User:"):
                role, content = "user", text.removeprefix("User:").strip()
            elif text.startswith("Assistant:"):
                role, content = "assistant", text.removeprefix("Assistant:").strip()
            else:
                raise ValueError(f"cannot parse source turn in {stage_key}: {text[:80]!r}")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unexpected role in {stage_key}: {role!r}")
        messages.append({"role": role, "content": content})
    return messages


def parse_transformed_stage(transformed_history: dict[str, Any], stage_id: str) -> list[dict[str, str]]:
    stage_key = source_stage_key(stage_id)
    raw_turns = transformed_history.get(stage_key)
    if not isinstance(raw_turns, list):
        raise ValueError(f"{stage_key} is missing or is not a list in transformed history")

    messages: list[dict[str, str]] = []
    for turn in raw_turns:
        if isinstance(turn, dict):
            role = str(turn.get("role") or "").strip().lower()
            content = str(turn.get("content") or "").strip()
        else:
            text = str(turn).strip()
            if text.startswith("User:"):
                role, content = "user", text.removeprefix("User:").strip()
            elif text.startswith("Assistant:"):
                role, content = "assistant", text.removeprefix("Assistant:").strip()
            else:
                raise ValueError(f"cannot parse transformed turn in {stage_key}: {text[:80]!r}")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unexpected role in transformed {stage_key}: {role!r}")
        messages.append({"role": role, "content": content})
    return messages


def recall_transformed_path(
    *,
    recall_transformed_root: Path,
    topic: str,
    sample: str,
    world: str,
    stage_id: str,
) -> Path:
    sample_dir = recall_transformed_root / topic / sample
    stem = sample
    stage_local = sample_dir / f"{stem}.{world}.{stage_id}.transformed_history.json"
    if stage_local.exists():
        return stage_local
    full = sample_dir / f"{stem}.{world}.transformed_history.json"
    if full.exists():
        return full
    raise FileNotFoundError(
        f"no recall transformed history for topic={topic} sample={sample} world={world} stage={stage_id}"
    )


def load_recall_transformed_stage(
    *,
    recall_transformed_root: Path,
    topic: str,
    sample: str,
    world: str,
    stage_id: str,
) -> tuple[list[dict[str, str]], str]:
    path = recall_transformed_path(
        recall_transformed_root=recall_transformed_root,
        topic=topic,
        sample=sample,
        world=world,
        stage_id=stage_id,
    )
    transformed_history = load_json(path)
    return parse_transformed_stage(transformed_history, stage_id), str(path)


def find_target_user_index(messages: list[dict[str, str]], target_user_turn: str) -> int:
    needle = " ".join(target_user_turn.strip().split())
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = " ".join(str(msg.get("content") or "").strip().split())
        if content == needle or needle in content or content in needle:
            return idx
    raise ValueError("target user turn not found")


def remove_target_block(messages: list[dict[str, str]], item: dict[str, Any]) -> list[dict[str, str]]:
    idx = find_target_user_index(messages, item["target_user_turn"])
    end = idx + 2 if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant" else idx + 1
    return deepcopy(messages[:idx] + messages[end:])


def build_no_store_context(source_messages: list[dict[str, str]], item: dict[str, Any]) -> list[dict[str, str]]:
    messages = deepcopy(source_messages)
    idx = find_target_user_index(messages, item["target_user_turn"])
    group = TEMPLATES.get("no_store", {})
    seed = str(item.get("id") or item.get("target_turn_id") or item.get("target_user_turn") or "")
    user_text = pick_template(group.get("user_suffix", []), "application", "no_store", seed)
    assistant_ack = pick_template(group.get("assistant_ack", []), "application", "no_store", seed)
    messages[idx]["content"] = append_sentence(messages[idx]["content"], user_text)
    if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant":
        messages[idx + 1]["content"] = merge_ack(messages[idx + 1]["content"], assistant_ack)
    return messages


def insertion_after_target_block(messages: list[dict[str, str]], item: dict[str, Any]) -> int:
    idx = find_target_user_index(messages, item["target_user_turn"])
    return idx + 2 if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant" else idx + 1


def candidate_pair_boundaries(messages: list[dict[str, str]], *, start_at: int) -> list[int]:
    """Return insertion points before later user turns, with end as fallback."""
    boundaries = [
        idx
        for idx in range(start_at, len(messages))
        if messages[idx].get("role") == "user"
    ]
    if not boundaries:
        boundaries.append(len(messages))
    return boundaries


def choose_boundary(boundaries: list[int], *seed_parts: str) -> int:
    if not boundaries:
        raise ValueError("boundaries must be non-empty")
    return boundaries[stable_index(*seed_parts) % len(boundaries)]


def target_reference(item: dict[str, Any]) -> str:
    explicit = str(item.get("forget_reference") or item.get("target_reference") or "").strip()
    if explicit:
        return explicit.rstrip(".!?")

    info = " ".join(str(item.get("unique_long_term_info") or item.get("lasting_memory") or "").split())
    info = info.rstrip(".!?")
    if not info:
        return "the earlier application-relevant request"

    lowered = info.lower()
    match = re.match(r"^the user has an? ([^,.;]+)", info, flags=re.IGNORECASE)
    if match:
        return f"the {match.group(1).strip()}"
    match = re.match(r"^the user is ([^,.;]+)", info, flags=re.IGNORECASE)
    if match:
        return f"the {match.group(1).strip()} preference"
    match = re.match(r"^for ([^,.;]+), the user prefers ", info, flags=re.IGNORECASE)
    if match:
        return f"the {match.group(1).strip()} preference"
    if lowered.startswith("the user prefers "):
        phrase = re.sub(r"^the user prefers\s+", "", info, flags=re.IGNORECASE)
        return f"the preference for {phrase}".rstrip(".!?")
    if lowered.startswith("the user wants "):
        phrase = re.sub(r"^the user wants\s+", "", info, flags=re.IGNORECASE)
        return f"the plan to {phrase}".rstrip(".!?")
    if lowered.startswith("the user "):
        phrase = re.sub(r"^the user\s+", "", info, flags=re.IGNORECASE)
        return f"the earlier detail that they {phrase}".rstrip(".!?")
    return info


def application_key_turn(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": str(item["target_turn_id"]),
        "stage_id": str(item["stage_id"]),
        "user_turn": str(item["target_user_turn"]),
        "task_goal": str(item.get("question") or item.get("unique_long_term_info") or ""),
        "key_phrase": target_reference(item),
    }


def build_shared_transform_context(
    conversation_package: dict[str, Any],
    item: dict[str, Any],
    world: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stage_id = str(item["stage_id"])
    stage_key = source_stage_key(stage_id)
    key_turn = application_key_turn(item)
    if world == "no_store":
        transformed = apply_no_store(
            conversation_package,
            period=stage_key,
            key_timestamp=str(item["target_turn_id"]),
            user_turn=str(item["target_user_turn"]),
        )
        metadata = dict(transformed.get(MEMORY_CONTROL_METADATA_KEY, {}))
        metadata["transform_version"] = "application_shared_stage_v1"
        metadata["no_store_insertions"] = [
            {
                "key_timestamp": str(item["target_turn_id"]),
                "key_stage": stage_key,
                "placement": "key_turn_suffix",
            }
        ]
        metadata["no_store_policy"] = {
            "placement": "same_user_turn_as_key",
            "ask_position": "end_of_stage",
            "stage_id": stage_id,
            "stage_key": stage_key,
            "source_logic": "memory_control_tests.transforms.apply_no_store",
        }
        transformed[MEMORY_CONTROL_METADATA_KEY] = metadata
    elif world == "forget":
        transformed = apply_stage_local_forget(
            conversation_package,
            key_turns=[key_turn],
            target_references=[target_reference(item)],
            stage_id=stage_id,
        )
    elif world == "no_use_active":
        transformed = apply_stage_local_no_use(
            conversation_package,
            key_turns=[key_turn],
            probe_turns=[],
            stage_id=stage_id,
            release=False,
        )
    elif world == "no_use_release":
        transformed = apply_stage_local_no_use(
            conversation_package,
            key_turns=[key_turn],
            probe_turns=[],
            stage_id=stage_id,
            release=True,
        )
        transformed = ensure_application_release(transformed, item)
    else:
        raise KeyError(world)

    return parse_transformed_stage(transformed, stage_id), {
        "control_source": "shared_transform",
        "transform_logic": "memory_control_tests.transforms",
        "memory_control_metadata": transformed.get(MEMORY_CONTROL_METADATA_KEY, {}),
    }


def ensure_application_release(transformed: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Guarantee that application no_use_release has a release before the MCQ.

    The recall stage-local rule keeps natural-history turns after release for
    utility probing. Application asks one MCQ after the stage context, so when a
    target appears too late for the stricter recall gap, appending the release
    pair at stage end is the least invasive way to make the released world
    actually released before the MCQ.
    """
    metadata = transformed.get(MEMORY_CONTROL_METADATA_KEY, {})
    insertions = metadata.get("no_use_insertions") if isinstance(metadata, dict) else None
    if not insertions:
        return transformed
    insertion = insertions[0]
    if "release_user_line_index" in insertion:
        return transformed

    stage_id = str(item["stage_id"])
    stage_key = source_stage_key(stage_id)
    lines = transformed.get(stage_key)
    if not isinstance(lines, list):
        return transformed

    seed = str(item.get("id") or item.get("target_turn_id") or item.get("target_user_turn") or "")
    group = TEMPLATES.get("no_use", {})
    release_user = pick_template(group.get("release_user", []), "application", "no_use_release_append", seed)
    release_assistant = pick_template(group.get("release_assistant", []), "application", "no_use_release_append", seed)
    release_line_index = len(lines)
    lines.extend([f"User: {release_user}", f"Assistant: {release_assistant}"])
    transformed[stage_key] = lines

    insertion.update(
        {
            "release_user_line_index": release_line_index,
            "release_assistant_line_index": release_line_index + 1,
            "release_insertion_mode": "append_to_stage_end_for_application_mcq",
            "release_user_line": release_user,
            "release_assistant_line": release_assistant,
        }
    )
    insertion.pop("release_skipped_reason", None)
    policy = metadata.get("no_use_policy", {})
    if isinstance(policy, dict):
        policy["application_release_fallback"] = "append_to_stage_end_before_mcq"
    return transformed


def build_forget_context(source_messages: list[dict[str, str]], item: dict[str, Any]) -> list[dict[str, str]]:
    messages = deepcopy(source_messages)
    insert_at = insertion_after_target_block(messages, item)
    reference = target_reference(item)
    group = TEMPLATES.get("forget", {})
    seed = str(item.get("id") or item.get("target_turn_id") or reference)
    user_template = pick_template(group.get("user", []), "application", "forget", seed)
    assistant_template = pick_template(group.get("assistant", []), "application", "forget", seed)
    forget_pair = [
        {"role": "user", "content": fill_template(user_template, target_reference=reference)},
        {"role": "assistant", "content": fill_template(assistant_template, target_reference=reference)},
    ]
    return messages[:insert_at] + forget_pair + messages[insert_at:]


def build_no_use_context(source_messages: list[dict[str, str]], item: dict[str, Any], *, release: bool) -> list[dict[str, str]]:
    messages = deepcopy(source_messages)
    earliest_insert_at = insertion_after_target_block(messages, item)
    group = TEMPLATES.get("no_use", {})
    seed = str(item.get("id") or item.get("target_turn_id") or item.get("target_user_turn") or "")
    restrict_boundaries = candidate_pair_boundaries(messages, start_at=earliest_insert_at)
    insert_at = choose_boundary(restrict_boundaries, "application", "no_use_restrict_position", seed)
    no_use_pair = [
        {
            "role": "user",
            "content": pick_template(group.get("restrict_user", []), "application", "no_use_restrict", seed),
        },
        {
            "role": "assistant",
            "content": pick_template(
                group.get("restrict_assistant", []), "application", "no_use_restrict", seed
            ),
        },
    ]
    messages = messages[:insert_at] + no_use_pair + messages[insert_at:]
    if release:
        release_start_at = insert_at + len(no_use_pair)
        release_boundaries = candidate_pair_boundaries(messages, start_at=release_start_at)
        release_at = choose_boundary(release_boundaries, "application", "no_use_release_position", seed)
        release_pair = [
            {
                "role": "user",
                "content": pick_template(group.get("release_user", []), "application", "no_use_release", seed),
            },
            {
                "role": "assistant",
                "content": pick_template(
                    group.get("release_assistant", []), "application", "no_use_release", seed
                ),
            },
        ]
        messages = messages[:release_at] + release_pair + messages[release_at:]
    return messages


def context_for_world(
    item: dict[str, Any],
    conversation_package: dict[str, Any],
    source_messages: list[dict[str, str]],
    world: str,
    *,
    control_source: str,
    recall_transformed_root: Path,
    sample: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if world == "never_seen_baseline":
        return remove_target_block(source_messages, item), {"control_source": "target_block_removed"}
    if world == "seen_baseline":
        return deepcopy(source_messages), {"control_source": "source_stage"}

    if control_source == "shared_transform":
        return build_shared_transform_context(conversation_package, item, world)

    if control_source == "recall_transformed":
        context, path = load_recall_transformed_stage(
            recall_transformed_root=recall_transformed_root,
            topic=str(item["topic"]),
            sample=sample,
            world=world,
            stage_id=str(item["stage_id"]),
        )
        return context, {
            "control_source": "recall_transformed_history",
            "transformed_history_path": path,
        }

    if control_source != "application":
        raise ValueError(f"unsupported control_source: {control_source}")
    if world == "no_store":
        return build_no_store_context(source_messages, item), {"control_source": "application_inline"}
    if world == "forget":
        return build_forget_context(source_messages, item), {"control_source": "application_inline"}
    if world == "no_use_active":
        return build_no_use_context(source_messages, item, release=False), {"control_source": "application_inline"}
    if world == "no_use_release":
        return build_no_use_context(source_messages, item, release=True), {"control_source": "application_inline"}
    raise KeyError(world)


def expected_for_world(item: dict[str, Any], world: str) -> str:
    if world in {"seen_baseline", "no_use_release"}:
        return item["expected"]["with_target_baseline"]
    if world in {"never_seen_baseline", "no_store", "forget", "no_use_active"}:
        return item["expected"]["without_target_baseline"]
    raise KeyError(world)


def item_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise TypeError("items.json must be a list or an object with an items list")


def load_topic_items(items_root: Path, topic: str, sample: str) -> list[dict[str, Any]]:
    sample_dir = items_root / topic / sample
    for name in ("items.json", "application_items.json"):
        path = sample_dir / name
        if path.exists():
            return item_rows(load_json(path))
    raise FileNotFoundError(f"no items.json or application_items.json under {sample_dir}")


def build_world_records(
    *,
    items_by_topic: dict[str, list[dict[str, Any]]],
    conversation_by_topic: dict[str, dict[str, Any]],
    control_source: str,
    recall_transformed_root: Path,
    sample: str,
) -> dict[str, Any]:
    all_items: list[dict[str, Any]] = []
    by_world: dict[str, list[dict[str, Any]]] = {world: [] for world in WORLD_ORDER}
    topic_counts = {topic: len([item for item in items if item.get("status") == "kept"]) for topic, items in items_by_topic.items()}

    for topic, items in items_by_topic.items():
        pmsgs = persona_messages(conversation_by_topic[topic])
        for item in items:
            if item.get("status") != "kept":
                continue
            source_messages = parse_source_stage(conversation_by_topic[topic], item["stage_id"])
            mcq_prompt = build_eval_prompt(item["question"], item["choices"])
            world_records: dict[str, dict[str, Any]] = {}
            for world in WORLD_ORDER:
                context_messages, control_metadata = context_for_world(
                    item,
                    conversation_by_topic[topic],
                    source_messages,
                    world,
                    control_source=control_source,
                    recall_transformed_root=recall_transformed_root,
                    sample=sample,
                )
                messages = pmsgs + context_messages + [{"role": "user", "content": mcq_prompt}]
                record = {
                    "id": item["id"],
                    "topic": item["topic"],
                    "stage_id": item["stage_id"],
                    "target_turn_id": item["target_turn_id"],
                    "world": world,
                    "expected_choice": expected_for_world(item, world),
                    "expected_behavior": "use_memory" if world in {"seen_baseline", "no_use_release"} else "without_memory",
                    "question": item["question"],
                    "choices": item["choices"],
                    "choice_roles": item["choice_roles"],
                    "messages": messages,
                    "context_messages": context_messages,
                    "context_source": {
                        "conversation_root": "data/generated",
                        "stage_key": source_stage_key(item["stage_id"]),
                        "synthetic_context": False,
                        **control_metadata,
                    },
                }
                world_records[world] = record
                by_world[world].append(record)
            all_items.append(
                {
                    "id": item["id"],
                    "topic": item["topic"],
                    "stage_id": item["stage_id"],
                    "target_turn_id": item["target_turn_id"],
                    "target_user_turn": item["target_user_turn"],
                    "unique_long_term_info": item["unique_long_term_info"],
                    "worlds": world_records,
                }
            )

    return {
        "schema_version": "application_worlds_v1",
        "world_order": WORLD_ORDER,
        "control_source": control_source,
        "topic_counts": topic_counts,
        "total_items": len(all_items),
        "total_world_records": sum(len(records) for records in by_world.values()),
        "items": all_items,
        "_by_world": by_world,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items-root",
        required=True,
        help="Directory containing <topic>/<sample>/items.json or application_items.json",
    )
    parser.add_argument("--conversation-root", default="data/generated", help="Directory containing <topic>/<sample>/conversation_package.json")
    parser.add_argument(
        "--control-source",
        choices=["shared_transform", "recall_transformed", "application"],
        default="shared_transform",
        help=(
            "Where non-baseline control worlds come from. The default reuses "
            "the same transform functions as recall, but targets application turns."
        ),
    )
    parser.add_argument(
        "--recall-transformed-root",
        default="data/recall/mcq_work",
        help="Root containing <topic>/<sample>/<sample>.<world>[.<stage>].transformed_history.json",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample", default="persona0_sample0")
    parser.add_argument("--topics", nargs="+", default=["travelPlanning", "financialConsultation", "medicalConsultation"])
    args = parser.parse_args()

    items_root = Path(args.items_root)
    conversation_root = Path(args.conversation_root)
    output_root = Path(args.output_root)
    recall_transformed_root = Path(args.recall_transformed_root)

    items_by_topic = {topic: load_topic_items(items_root, topic, args.sample) for topic in args.topics}
    conversation_by_topic = {
        topic: load_json(conversation_root / topic / args.sample / "conversation_package.json")
        for topic in args.topics
    }
    result = build_world_records(
        items_by_topic=items_by_topic,
        conversation_by_topic=conversation_by_topic,
        control_source=args.control_source,
        recall_transformed_root=recall_transformed_root,
        sample=args.sample,
    )
    by_world = result.pop("_by_world")

    dump_json(output_root / "application_worlds.json", result)
    for world, records in by_world.items():
        dump_json(output_root / "plain_inputs" / f"{world}.json", {"world": world, "items": records})
    dump_json(
        output_root / "README.json",
        {
            "description": "Per-stage application transformed worlds rendered as plain-eval messages.",
            "worlds": WORLD_ORDER,
            "expected": {
                "never_seen_baseline": "without_memory",
                "seen_baseline": "use_memory",
                "no_store": "without_memory",
                "forget": "without_memory",
                "no_use_active": "without_memory",
                "no_use_release": "use_memory",
            },
            "topic_counts": result["topic_counts"],
            "total_items": result["total_items"],
        },
    )
    print(json.dumps({"total_items": result["total_items"], "worlds": {k: len(v) for k, v in by_world.items()}}, indent=2))


if __name__ == "__main__":
    main()
