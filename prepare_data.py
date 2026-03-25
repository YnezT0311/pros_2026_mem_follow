from tqdm import tqdm
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import json
import random
import torch
import ast
import yaml
import argparse
import sys
import ast
import traceback
import hashlib
from json_repair import repair_json

from query_llm import QueryLLM
import utils

GENERAL_HISTORY_SECTION_NAMES = [
    "General Personal History Initial Stage",
    "General Personal History Early Stage",
    "General Personal History Intermediate Stage",
    "General Personal History Late Stage",
]

CONTEXTUAL_HISTORY_SECTION_NAMES = [
    "Contextual Personal History Initial Stage",
    "Contextual Personal History Early Stage",
    "Contextual Personal History Intermediate Stage",
    "Contextual Personal History Late Stage",
]

EVENT_HISTORY_SECTION_NAMES = [
    "Event History Initial Stage",
    "Event History Early Stage",
    "Event History Intermediate Stage",
    "Event History Late Stage",
]

INTERACTION_SOURCE_SECTION_NAMES = [
    "Interaction Source Dates Initial Stage",
    "Interaction Source Dates Early Stage",
    "Interaction Source Dates Intermediate Stage",
    "Interaction Source Dates Late Stage",
]

INTERACTION_HISTORY_SECTION_NAMES = [
    "Interaction History Initial Stage",
    "Interaction History Early Stage",
    "Interaction History Intermediate Stage",
    "Interaction History Late Stage",
]

CONVERSATION_HISTORY_SECTION_NAMES = [
    "Conversation History Initial Stage",
    "Conversation History Early Stage",
    "Conversation History Intermediate Stage",
    "Conversation History Late Stage",
]

CONVERSATION_SECTION_NAMES = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]


TQDM_DISABLE = not sys.stderr.isatty()

TOPIC_SENSITIVE_POOL_HINTS = {
    # TODO: This mapping currently covers only the baseline topics available in data/output.
    # If more topics are added later, extend TOPIC_SENSITIVE_POOL_HINTS and
    # the information-type inference logic together so recurring sensitive anchors stay aligned.
    "financialConsultation": {
        "named_contact": ["family bookkeeper", "bank representative", "community workshop organizer"],
        "account_or_balance": ["checking account", "credit-card balance", "shared family ledger"],
        "document_or_record_reference": ["spreadsheet snapshot", "receipt archive", "payment log"],
    },
    "legalConsultation": {
        "named_contact": ["clinic coordinator", "junior associate", "platform takedown contact"],
        "legal_dispute_detail": ["tenant file", "clause library audit", "impersonation complaint"],
        "document_or_record_reference": ["intake sheet", "draft agreement", "audio transcript"],
    },
    "medicalConsultation": {
        "named_contact": ["clinic coordinator", "mentor clinician", "language-assessment contact"],
        "medical_symptom": ["vestibular migraine pattern", "sensory overload trigger"],
        "document_or_record_reference": ["study summary", "clinical note template", "prototype log export"],
    },
    "travelPlanning": {
        "named_contact": ["guesthouse contact", "residency coordinator", "cargo liaison"],
        "private_schedule": ["departure window", "residency calendar", "cargo handoff slot"],
        "document_or_record_reference": ["lodging note", "route note", "packing checklist"],
    },
}


def _history_items_in_order(history_dict):
    if not isinstance(history_dict, dict):
        return []
    items = []
    for date, record in history_dict.items():
        if isinstance(record, dict):
            items.append((date, record))
    items.sort(key=lambda x: x[0])
    return items


def build_sensitive_info_pool(persona_text, pii_profile, topic):
    pool = {}
    if isinstance(pii_profile, dict):
        if pii_profile.get("email"):
            pool["email"] = [pii_profile["email"]]
        if pii_profile.get("phone"):
            pool["phone_number"] = [pii_profile["phone"]]
        if pii_profile.get("address"):
            pool["address"] = [pii_profile["address"]]
        if pii_profile.get("id"):
            pool["synthetic_id"] = [pii_profile["id"]]

    text = persona_text or ""
    name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    persona_name = name_match.group(1) if name_match else "Alex"
    seed = hashlib.md5(f"{topic}|{persona_name}|{text}".encode("utf-8")).hexdigest()[:6].upper()
    hints = TOPIC_SENSITIVE_POOL_HINTS.get(topic, {})

    def ensure_list(key, values):
        if not values:
            return
        pool.setdefault(key, [])
        for value in values:
            if value not in pool[key]:
                pool[key].append(value)

    ensure_list("named_contact", [f"{persona_name}'s trusted contact", f"{persona_name}'s project coordinator"])
    if "booking_identifier" in hints:
        ensure_list("booking_identifier", [f"{seed}-BOOK", f"{seed}-REF"])
    if "account_or_balance" in hints:
        ensure_list("account_or_balance", hints["account_or_balance"])
    if "legal_dispute_detail" in hints:
        ensure_list("legal_dispute_detail", hints["legal_dispute_detail"])
    if "medical_symptom" in hints:
        ensure_list("medical_symptom", hints["medical_symptom"])
    if "private_schedule" in hints:
        ensure_list("private_schedule", hints["private_schedule"])
    if "document_or_record_reference" in hints:
        ensure_list("document_or_record_reference", hints["document_or_record_reference"])
    if "named_contact" in hints:
        ensure_list("named_contact", hints["named_contact"])

    return pool


def _infer_information_types(topic, event_text):
    event_lower = (event_text or "").lower()
    info_types = []

    if any(k in event_lower for k in ["email", "e-mail"]):
        info_types.append("email")
    if any(k in event_lower for k in ["phone", "call", "text number", "voicemail"]):
        info_types.append("phone_number")
    if any(k in event_lower for k in ["address", "street", "unit", "apartment"]):
        info_types.append("address")
    if any(k in event_lower for k in ["account", "balance", "debt", "payment", "credit", "ledger", "receipt", "cash-flow"]):
        info_types.append("account_or_balance")
    if any(k in event_lower for k in ["booking", "confirmation", "ticket", "reservation", "flight", "voucher", "pass"]):
        info_types.append("document_or_record_reference")
    if any(k in event_lower for k in ["route", "itinerary", "connection", "departure window", "arrival window"]):
        info_types.append("private_schedule")
    if any(k in event_lower for k in ["symptom", "migraine", "diagnosis", "dosing", "medication", "clinical", "patient"]):
        info_types.append("medical_symptom")
    if any(k in event_lower for k in ["dose", "dosing", "medication", "prescription"]):
        info_types.append("medication_or_dosing")
    if any(k in event_lower for k in ["contract", "clause", "evidence", "clinic", "landlord", "tenant", "impersonation", "deepfake", "notice"]):
        info_types.append("legal_dispute_detail")
    if any(k in event_lower for k in ["cousin", "neighbor", "friend", "relative", "family", "aunt", "mentor", "collaborator"]):
        info_types.append("family_or_relationship_detail")
        info_types.append("named_contact")
    if any(k in event_lower for k in ["calendar", "schedule", "timeline", "itinerary", "shift", "workshop", "slot"]):
        info_types.append("private_schedule")
    if any(k in event_lower for k in ["file", "record", "archive", "log", "report", "worksheet", "spreadsheet", "notebook"]):
        info_types.append("document_or_record_reference")

    if not info_types:
        default_by_topic = {
            "financialConsultation": "account_or_balance",
            "legalConsultation": "legal_dispute_detail",
            "medicalConsultation": "medical_symptom",
            "travelPlanning": "private_schedule",
        }
        info_types.append(default_by_topic.get(topic, "document_or_record_reference"))

    deduped = []
    seen = set()
    for item in info_types:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _project_sensitive_info(event_text, sensitive_info_pool, topic):
    info = {}
    for key in _infer_information_types(topic, event_text):
        values = list((sensitive_info_pool or {}).get(key, []))
        if values:
            info[key] = values[:2]
    return info


def derive_interaction_metadata(LLM, topic, record, index, sensitive_info_pool, persona_text, general_history, verbose=False):
    response = LLM.query_llm(
        step='derive_interaction_details',
        topic=topic,
        data={
            'event_record': record,
            'sensitive_info_pool': sensitive_info_pool or {},
            'persona': persona_text,
            'general_history': general_history or {},
        },
        verbose=verbose,
    )
    match = re.search(r'```(?:json|python|plaintext)?\s*(.*?)\s*```', response, re.DOTALL)
    response = match.group(1) if match else response
    parsed = json.loads(repair_json(response))

    task_goal = str(parsed.get("task_goal", "")).strip()
    if not task_goal:
        raise RuntimeError(
            f"Interaction detail derivation missing task_goal for topic={topic}, event_id={record.get('event_id')}: {parsed!r}"
        )

    context_can_add = parsed.get("context_can_add")
    if not isinstance(context_can_add, dict):
        raise RuntimeError(
            f"Interaction detail derivation missing dict context_can_add for topic={topic}, event_id={record.get('event_id')}: {parsed!r}"
        )
    cleaned_context = {}
    for k, v in list(context_can_add.items())[:5]:
        key = str(k).strip()
        val = str(v).strip()
        if key and val:
            cleaned_context[key] = val
    if len(cleaned_context) < 1:
        raise RuntimeError(
            f"Interaction detail derivation produced empty context_can_add for topic={topic}, event_id={record.get('event_id')}: {parsed!r}"
        )

    selected_sensitive_info = parsed.get("sensitive_info")
    if not isinstance(selected_sensitive_info, dict):
        raise RuntimeError(
            f"Interaction detail derivation missing dict sensitive_info for topic={topic}, event_id={record.get('event_id')}: {parsed!r}"
        )
    cleaned_info = {}
    for k, v in selected_sensitive_info.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()][:3]
        else:
            vals = [str(v).strip()] if str(v).strip() else []
        if vals:
            cleaned_info[key] = vals

    return {
        "task_goal": task_goal,
        "context_can_add": cleaned_context,
        "sensitive_info": cleaned_info,
    }


def build_event_history(history_dict, topic, period_key, sensitive_info_pool):
    items = _history_items_in_order(history_dict)
    if not items:
        return {}

    event_history = {}
    date_to_event_id = {}
    for idx, (date, record) in enumerate(items):
        event_text = record.get("Event", "")
        has_change = bool(record.get("[Old Event]") or record.get("Old Event") or record.get("[Reasons of Change]") or record.get("Reasons of Change"))
        subtype = "change" if has_change else "init"
        event_id = f"E_{period_key.upper()}_{idx + 1:03d}"
        date_to_event_id[date] = event_id
        event_history[date] = {
            **record,
            "event_id": event_id,
            "turn_type": "update",
            "update_subtype": subtype,
            "sensitive_info": _project_sensitive_info(
                " ".join(
                    str(record.get(k, ""))
                    for k in ["Event", "[Old Event]", "[Reasons of Change]", "Anchors"]
                ),
                sensitive_info_pool,
                topic,
            ),
            "relations": [],
        }

    for date, record in event_history.items():
        old_date = record.get("[Old Event Date]") or record.get("Old Event Date")
        if old_date and old_date in date_to_event_id:
            record["relations"].append({
                "type": "evolves_from",
                "source_event_id": date_to_event_id[old_date],
            })

    return event_history


def select_interaction_dates(LLM, event_history, topic, period_key, verbose=False):
    items = _history_items_in_order(event_history)
    if not items:
        return []

    target_count = {
        "init": 8,
        "week": 4,
        "month": 4,
        "year": 4,
    }.get(period_key, max(1, len(items) // 4))
    target_count = min(target_count, len(items))

    response = LLM.query_llm(
        step='select_interaction_events',
        topic=topic,
        data={'event_history': event_history, 'target_count': target_count},
        verbose=verbose,
    )
    match = re.search(r'```(?:json|python|plaintext)?\s*(.*?)\s*```', response, re.DOTALL)
    response = match.group(1) if match else response
    parsed = json.loads(repair_json(response))
    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Interaction selection did not return a JSON list for topic={topic}, period={period_key}: {parsed!r}"
        )

    chosen = []
    valid_dates = {date for date, _ in items}
    invalid = []
    duplicates = []
    seen = set()
    for item in parsed:
        if item not in valid_dates:
            invalid.append(item)
            continue
        if item in seen:
            duplicates.append(item)
            continue
        seen.add(item)
        chosen.append(item)

    if invalid or duplicates or len(chosen) != target_count:
        raise RuntimeError(
            "Interaction selection failed for "
            f"topic={topic}, period={period_key}. "
            f"Expected {target_count} unique valid timestamps, got {len(chosen)}. "
            f"Invalid={invalid}. Duplicates={duplicates}. Parsed={parsed}."
        )
    return chosen


def build_interaction_history(LLM, event_history, topic, period_key, sensitive_info_pool, selected_dates, persona_text, general_history, verbose=False):
    items = _history_items_in_order(event_history)
    if not items:
        return {}

    interaction_history = {}
    selected_set = set(selected_dates)
    interaction_idx = 1
    for date, record in items:
        if date not in selected_set:
            continue
        base_event_id = record.get("event_id")
        meta = derive_interaction_metadata(
            LLM,
            topic,
            record,
            interaction_idx - 1,
            sensitive_info_pool,
            persona_text,
            general_history,
            verbose=verbose,
        )
        interaction_history[f"{date}#I{interaction_idx:02d}"] = {
            "event_id": f"I_{period_key.upper()}_{interaction_idx:03d}",
            "turn_type": "help_seek",
            "update_subtype": None,
            "timestamp": f"{date}-I{interaction_idx:02d}",
            "source_event_id": base_event_id,
            "source_event_date": date,
            "[Prev Event]": record.get("Event", ""),
            "[Task Goal]": meta["task_goal"],
            "[Context Can Add]": meta["context_can_add"],
            "[Sensitive Info]": meta["sensitive_info"],
            "relations": [{"type": "derived_from", "source_event_id": base_event_id}],
        }
        interaction_idx += 1

    return interaction_history


def build_conversation_history(event_history, interaction_history):
    conversation_history = []
    interaction_by_source = {}
    for _, interaction in _history_items_in_order(interaction_history):
        interaction_by_source.setdefault(interaction.get("source_event_date"), []).append(interaction)

    for date, record in _history_items_in_order(event_history):
        event_item = {
            "timestamp": date,
            "kind": "event",
            "event_id": record.get("event_id"),
            "turn_type": "update",
            "update_subtype": record.get("update_subtype"),
            "event": record.get("Event", ""),
            "category": record.get("Category"),
            "anchors": record.get("Anchors", {}),
            "[Old Event Date]": record.get("[Old Event Date]") or record.get("Old Event Date"),
            "[Old Event]": record.get("[Old Event]") or record.get("Old Event"),
            "[Reasons of Change]": record.get("[Reasons of Change]") or record.get("Reasons of Change"),
            "[Sensitive Info]": record.get("sensitive_info", {}),
            "relations": record.get("relations", []),
        }
        for key in (
            "[Fact] Likes",
            "[Fact] Dislikes",
            "[Old Fact] Likes",
            "[Old Fact] Dislikes",
            "[Updated Fact] Likes",
            "[Updated Fact] Dislikes",
        ):
            if key in record:
                event_item[key] = record.get(key)
        conversation_history.append(event_item)
        for interaction in interaction_by_source.get(date, []):
            conversation_history.append({
                "timestamp": interaction.get("timestamp", date),
                "kind": "interaction",
                **interaction,
            })
    return conversation_history


def is_retryable_error(e):
    err = str(e).lower()
    retry_signals = [
        "connection error",
        "temporarily",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
        "conversation/history date mismatch",
        "incomplete output",
    ]
    return any(s in err for s in retry_signals)


def get_missing_output_sections(data, topic):
    if topic in ("writing", "email"):
        required = ["Conversation"]
    else:
        required = CONVERSATION_SECTION_NAMES

    missing = []
    for key in required:
        value = data.get(key)
        if not isinstance(value, list) or not value:
            missing.append(key)
    return missing


def get_missing_conversation_prereq_sections(data, topic):
    if topic in ("writing", "email"):
        return []

    required = [
        "Original Persona",
        "Expanded Persona",
        "Topic",
        "Sensitive Info Pool",
        *GENERAL_HISTORY_SECTION_NAMES,
        *CONTEXTUAL_HISTORY_SECTION_NAMES,
        *CONVERSATION_HISTORY_SECTION_NAMES,
    ]
    missing = []
    for key in required:
        value = data.get(key)
        if value is None or value == {} or value == []:
            missing.append(key)
    return missing


def _conversation_log_dir(output_file_path):
    return output_file_path + ".logs"


def _safe_log_slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def maybe_write_conversation_log(output_file_path, args, stage_name, artifact_name, payload):
    if not args.get('inference', {}).get('save_output_response', False):
        return
    log_dir = _conversation_log_dir(output_file_path)
    os.makedirs(log_dir, exist_ok=True)
    stem = f"{_safe_log_slug(stage_name)}.{_safe_log_slug(artifact_name)}"
    path = os.path.join(log_dir, stem + (".json" if isinstance(payload, (dict, list)) else ".txt"))
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, (dict, list)):
            json.dump(payload, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(payload))


def is_output_complete(output_file_path, topic):
    if not os.path.exists(output_file_path):
        return False
    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return not get_missing_output_sections(data, topic)


def _extract_side_note_dates(conversation_lines):
    dates = []
    if not isinstance(conversation_lines, list):
        return dates
    for line in conversation_lines:
        if not isinstance(line, str):
            continue
        if not (line.startswith("Side_Note") or line.startswith("Side_Note:")):
            continue
        for m in re.findall(r"\b\d{2}/\d{2}/\d{4}(?:-I\d{2})?\b", line):
            dates.append(m)
    return dates


def _dedupe_side_note_blocks(conversation_lines):
    if not isinstance(conversation_lines, list):
        return conversation_lines
    out = []
    seen_dates = set()
    i = 0
    n = len(conversation_lines)
    while i < n:
        line = conversation_lines[i]
        if not isinstance(line, str) or not (line.startswith("Side_Note") or line.startswith("Side_Note:")):
            out.append(line)
            i += 1
            continue

        j = i + 1
        while j < n:
            nxt = conversation_lines[j]
            if isinstance(nxt, str) and (nxt.startswith("Side_Note") or nxt.startswith("Side_Note:")):
                break
            j += 1
        block = conversation_lines[i:j]

        dates = re.findall(r"\b\d{2}/\d{2}/\d{4}(?:-I\d{2})?\b", line)
        key = dates[0] if dates else f"__no_date_{i}"
        if key not in seen_dates:
            out.extend(block)
            seen_dates.add(key)
        i = j
    return out


def _assert_conversation_aligned(conversation_lines, expected_history_dict, label):
    side_note_dates = _extract_side_note_dates(conversation_lines)
    expected_items_by_timestamp = {}
    if isinstance(expected_history_dict, dict):
        iterable = expected_history_dict.items()
    elif isinstance(expected_history_dict, list):
        iterable = []
        for item in expected_history_dict:
            if isinstance(item, dict) and item.get("timestamp"):
                iterable.append((item["timestamp"], item))
    else:
        iterable = []

    for raw_date, item in iterable:
        timestamp = str(raw_date).split("#")[0]
        expected_items_by_timestamp.setdefault(timestamp, []).append(item)

    expected_dates = set(expected_items_by_timestamp.keys())
    invalid_dates = [d for d in side_note_dates if d not in expected_dates]
    if invalid_dates:
        sample = ", ".join(invalid_dates[:5])
        raise RuntimeError(
            f"Conversation/history date mismatch at {label}. "
            f"Found Side_Note dates not in expected history: {sample}"
        )
    counts = {}
    for d in side_note_dates:
        counts[d] = counts.get(d, 0) + 1
    mismatched_dates = [d for d in expected_dates if counts.get(d, 0) != 1]
    if mismatched_dates:
        sample = ", ".join(
            f"{d} expected 1 got {counts.get(d, 0)}"
            for d in mismatched_dates[:5]
        )
        raise RuntimeError(
            f"Conversation/history date mismatch at {label}. "
            f"Side_Note count mismatch: {sample}"
        )
    missing_dates = [d for d in expected_dates if d not in counts]
    if missing_dates:
        raise RuntimeError(
            f"Conversation/history date mismatch at {label}. "
            f"Missing Side_Note dates from expected history: {', '.join(missing_dates[:5])}"
        )


def rewrite_conversations_from_existing(LLM, existing_data, curr_topic, start_time, output_file_path, args):
    conversation_histories = [existing_data[name] for name in CONVERSATION_HISTORY_SECTION_NAMES]
    LLM.expanded_persona = existing_data["Expanded Persona"]
    LLM.init_personal_history = json.dumps(conversation_histories[0], ensure_ascii=False, indent=2)
    LLM.first_expand_personal_history = json.dumps(conversation_histories[1], ensure_ascii=False, indent=2)
    LLM.second_expand_personal_history = json.dumps(conversation_histories[2], ensure_ascii=False, indent=2)
    LLM.third_expand_personal_history = json.dumps(conversation_histories[3], ensure_ascii=False, indent=2)

    last_timestamps = []
    for section_name in GENERAL_HISTORY_SECTION_NAMES + CONTEXTUAL_HISTORY_SECTION_NAMES:
        last_timestamps.append(utils.extract_last_timestamp(existing_data[section_name]))
    last_timestamps = utils.merge_timestamps(last_timestamps)

    sensitive_info_pool = existing_data["Sensitive Info Pool"]
    steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
    data_names = CONVERSATION_SECTION_NAMES

    for conv_idx, (step, data_name) in enumerate(zip(steps, data_names)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        try:
            maybe_write_conversation_log(output_file_path, args, data_name, "expected_history", conversation_histories[conv_idx])
            response = LLM.query_llm(
                step=step,
                topic=curr_topic,
                idx_topic=0,
                start_time=start_time,
                verbose=args['inference']['verbose'],
                sensitive_info_pool=sensitive_info_pool,
            )
            maybe_write_conversation_log(output_file_path, args, data_name, "raw", response)
            reflected = LLM.query_llm(
                step='reflect_' + step,
                topic=curr_topic,
                data=response,
                action=1,
                verbose=args['inference']['verbose'],
            )
            maybe_write_conversation_log(output_file_path, args, data_name, "reflect_round1", reflected)
            response = LLM.query_llm(step='reflect_' + step, topic=curr_topic, action=2, verbose=args['inference']['verbose'])
            maybe_write_conversation_log(output_file_path, args, data_name, "reflect_round2", response)
            expanded_conversation = parse_conversation_sections(LLM, response, curr_topic, last_timestamps[conv_idx], verbose=args['inference']['verbose'])
            maybe_write_conversation_log(output_file_path, args, data_name, "parsed", expanded_conversation)
            expanded_conversation = _dedupe_side_note_blocks(expanded_conversation)
            maybe_write_conversation_log(output_file_path, args, data_name, "deduped", expanded_conversation)
            _assert_conversation_aligned(expanded_conversation, conversation_histories[conv_idx], data_name)
            utils.append_json_to_file(expanded_conversation, output_file_path, curr_data_name=data_name, parse_json=False, parse_list=False)
        except Exception as e:
            maybe_write_conversation_log(output_file_path, args, data_name, "error", repr(e))
            raise


def prepare_persona(LLM, idx_persona, all_personas, args):
    # Load a persona
    found = None if args['inference'].get('force_regen_persona', False) else utils.find_existing_persona_files(idx_persona)
    # found = False
    if found:
        # Ensure that every data file with the same idx_persona share the same persona
        persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year \
            = found['persona'], found['expanded_persona'], found['start_time'], found['init_general_personal_history'], found['general_personal_history_next_week'], found['general_personal_history_next_month'], found['general_personal_history_next_year']
        LLM.expanded_persona = expanded_persona
        if not start_time:
            start_time = utils.pick_a_random_time()
        if args['inference']['verbose']:
            print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}')
            print(persona)
            print(f'{utils.Colors.OKGREEN}{"Expanded Persona"}:{utils.Colors.ENDC}')
            print(expanded_persona)
        row_idx = utils.find_persona_row_index(persona, all_personas)
        if row_idx is not None:
            mapping = utils.load_persona_index_map()
            mapping.setdefault("source_row_to_persona_idx", {})[str(row_idx)] = int(idx_persona)
            utils.save_persona_index_map(mapping)
    else:
        # Create a new persona for the new idx_persona
        persona, row_idx, _ = utils.get_or_create_persona_for_index(idx_persona, all_personas)
        if args['inference']['verbose']:
            print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}{persona}')
            print(f'{utils.Colors.OKBLUE}Persona mapping:{utils.Colors.ENDC} idx={idx_persona} -> source_row_index={row_idx}')

        # Expand the persona to at least five sentences
        start_time = utils.pick_a_random_time()
        expanded_persona = LLM.query_llm(step='expand_persona', persona=persona, start_time=start_time, verbose=args['inference']['verbose'])
        init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year = None, None, None, None

    return persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year


def prepare_topics(idx_topic, all_topics, curr_topic, args):
    # Process each topic as needed
    print(f'{utils.Colors.OKBLUE}Processing topic: {curr_topic}, {idx_topic + 1}/{len(all_topics)}{utils.Colors.ENDC}')

    # Load a random conversation history from the chosen real-world dataset
    # TODO: figure out the source
    if curr_topic == 'writing':
        source_dir = args['datasets']['writing_source_dir']
    elif curr_topic == 'email':
        source_dir = args['datasets']['email_source_dir']
    elif curr_topic == 'legal':
        source_dir = args['datasets']['legal_source_dir']
    elif curr_topic == 'therapy':
        source_dir = args['datasets']['therapy_source_dir']
    else:
        source_dir = None
        # print(f'{utils.Colors.WARNING}No source data is available for the topic: {curr_topic}{utils.Colors.ENDC}')

    all_source_files = utils.load_all_source_data(source_dir, curr_topic) if source_dir is not None else None
    return source_dir, all_source_files


def parse_conversation_sections(LLM, input_conversation, topic, last_timestamp, verbose):
    """
    :param input_conversation: A list of strings representing the conversation
    We define each section in the conversation as a group of lines before the next Side_Note
    """
    def expand_section(LLM, section, last_timestamp):
        if verbose:
            print(f'{utils.Colors.OKGREEN}{"Original Section"}:{utils.Colors.ENDC}')
            print(section)

        response = LLM.query_llm(step='expand_conversation_section', topic=topic, data={'section': section, 'last_timestamp': last_timestamp}, verbose=False)
        match = re.search(r'```(?:python|plaintext)?\s*(.*?)\s*```', response, re.DOTALL)
        response = match.group(1) if match else response
        response = response.strip().replace('\n', '')
        if '=' in response:
            response = re.sub(r'^\s*\w+\s*=\s*', '', response, count=1).strip()
        if response[-1] != ']':
            response += ']'
        if response[-2] != '"' and response[-3] == '"':
            response = response[:-3] + '"]'

        if verbose:
            print(f'{utils.Colors.OKGREEN}{"Expanded Section"}:{utils.Colors.ENDC}')
            print(response)

        # response = ast.literal_eval(response)
        response = repair_json(response)
        response = json.loads(response)

        if verbose:
            print('Parsed section', response, '\n\n')
        return response

    # Keywords to identify the start of a new section
    keywords = {'Side_Note', 'Side_Notes', '[Side_Note]', '[Side_Notes]', 'Side', '[Side'}
    sections = []  # To store the parsed sections
    with_next_sidenote = []
    current_section = []  # To collect strings for the current section

    # print('input_conversation', input_conversation, '\n\n')
    match = re.search(r'```(?:python|plaintext)?\s*(.*?)\s*```', input_conversation, re.DOTALL)
    input_conversation = match.group(1) if match else input_conversation
    input_conversation = input_conversation.strip().replace('\n', '')
    if '=' in input_conversation:
        input_conversation = re.sub(r'^\s*\w+\s*=\s*', '', input_conversation, count=1).strip()
    if input_conversation[-1] != ']':
        input_conversation += ']'
    if verbose:
        print('parsed input_conversation', input_conversation, '\n\n')
    # input_conversation = input_conversation.strip("```python").strip("```plaintext").strip()

    input_conversation = repair_json(input_conversation)
    input_conversation = json.loads(input_conversation)
    # input_conversation = ast.literal_eval(input_conversation)
    # print('input_conversation', input_conversation, '\n\n')

    for idx, line in enumerate(input_conversation):
        # Check if the line starts with any of the keywords
        if any(line.startswith(keyword) for keyword in keywords):
            # Save the current section (if not empty) and start a new one
            if current_section:
                # # Add the next line containing the next Side_Note, if any, to support smoother transition
                # if idx + 1 < len(input_conversation):
                #     current_section.append(input_conversation[idx + 1])
                sections.append(current_section)
                current_section = []
        # Add the current line to the current section
        current_section.append(line)

    # Add the last section if there is one
    if current_section:
        sections.append(current_section)
    # print('all sections', sections, '\n\n')

    expanded_conversation = []
    for idx, section in enumerate(sections):
        # print('section', section, '\n\n')
        if section and not any(isinstance(line, str) and any(line.startswith(keyword) for keyword in keywords) for line in section):
            # Keep a plain intro section as-is. The downstream section expander assumes
            # a Side_Note-oriented block template and would otherwise invent a fake
            # timestamped Side_Note for this opening.
            expanded_section = section
        else:
            expanded_section = expand_section(LLM, section, last_timestamp)

        expanded_conversation += expanded_section

    if verbose:
        print(f'{utils.Colors.OKGREEN}{"Expanded Conversation"}:{utils.Colors.ENDC}')
        print(expanded_conversation)

    return expanded_conversation


def prepare_data_on_writing_topic(LLM, topic, persona, source_data, output_file_path, args):
    # Convert the writing sample into a conversation
    preferences = LLM.query_llm(step='prepare_new_content', data=persona, action='preferences', data_type=topic, verbose=args['inference']['verbose'])
    if topic == 'email':
        source_data = LLM.query_llm(step='rewrite_email', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])
    elif topic == 'writing':
        source_data = LLM.query_llm(step='rewrite_creative_writing', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])

    updated_writing_sample = LLM.query_llm(step='prepare_new_content', data=source_data, action='rewrite_from_persona', data_type=topic, verbose=args['inference']['verbose'])
    if 'python' in preferences or 'plaintext' in preferences:
        preferences = preferences.strip("```python").strip("```plaintext").strip()
    if 'plaintext' in updated_writing_sample:
        updated_writing_sample = updated_writing_sample.strip("```plaintext").strip()

    conversation = LLM.query_llm(step='prepare_new_content', action='rewrite_as_conversation', data_type=topic, verbose=args['inference']['verbose'])
    if conversation.startswith('```python'):
        conversation = conversation.replace('```python', '', 1)
    conversation = conversation.strip("```plaintext")
    try:
        conversation = json.loads(conversation)
    except:
        conversation = conversation

    # if 'python' in conversation or 'plaintext' in conversation:
    #     conversation = conversation.strip("```plaintext").replace('```python', '', 1).strip()
    #     conversation = ast.literal_eval(conversation)
    # # conversation.append("User: Could you please help me write another sample?")

    responses = [source_data, preferences, updated_writing_sample, conversation]
    data_names = ['Original Sample', 'Writing and Formatting Styles', 'Updated Writing Sample', 'Conversation']
    for response, data_name in zip(responses, data_names):
        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)


def prepare_data_on_other_topics(LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                 init_general_personal_history, first_expand_general_personal_history, second_expand_general_personal_history, third_expand_general_personal_history, args):
    # Feed the thread with a seeding data from the real-world conversation
    if source_dir is not None:
        source_conversation = utils.preprocess_source_data(source_data, curr_topic)
        _ = LLM.query_llm(step='source_data', seed=source_conversation, verbose=args['inference']['verbose'])
    else:
        _ = LLM.query_llm(step='elaborate_topic', topic=curr_topic, verbose=args['inference']['verbose'])

    # Generate general and contextual personal histories across time frames
    steps = ['init_general_personal_history', 'first_expand_general_personal_history', 'second_expand_general_personal_history', 'third_expand_general_personal_history',
             'init_contextual_personal_history', 'first_expand_contextual_personal_history', 'second_expand_contextual_personal_history', 'third_expand_contextual_personal_history']
    data_names = GENERAL_HISTORY_SECTION_NAMES + CONTEXTUAL_HISTORY_SECTION_NAMES
    existing_general_personal_history = {'init_general_personal_history': init_general_personal_history, 'first_expand_general_personal_history': first_expand_general_personal_history,
                                         'second_expand_general_personal_history': second_expand_general_personal_history, 'third_expand_general_personal_history': third_expand_general_personal_history}
    # steps = ['init_general_personal_history', 'init_contextual_personal_history']
    # data_names = ['Init General Personal History', 'Init Contextual Personal History']
    # existing_general_personal_history = {'init_general_personal_history': LLM.init_general_personal_history}

    last_timestamps = []
    normalized_history = {}
    for step, data_name in tqdm(zip(steps, data_names), disable=TQDM_DISABLE):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        # Only generate general personal history once, to be shared across multiple topics for the same persona
        # if idx_topic > 0 and step in existing_general_personal_history:
        #     utils.append_json_to_file(existing_general_personal_history[step], output_file_path, curr_data_name=data_name, parse_json=True)
        #     continue
        if step in existing_general_personal_history:
            if existing_general_personal_history[step] is not None:
                if step == 'init_general_personal_history':
                    print('Loading existing general personal history.')
                # Use existing general personal history shared across multiple topics for the same persona
                utils.append_json_to_file(existing_general_personal_history[step], output_file_path, curr_data_name=data_name, parse_json=True)
                normalized_history[step] = existing_general_personal_history[step]
                last_timestamps.append(utils.extract_last_timestamp(existing_general_personal_history[step]))
                continue
            else:
                if step == 'init_general_personal_history':
                    print('Generating new general personal history.')

        response = LLM.query_llm(step=step, persona=expanded_persona, topic=curr_topic, idx_topic=idx_topic, start_time=start_time, verbose=args['inference']['verbose'])

        if not isinstance(response, str):
            raise RuntimeError(f"LLM returned non-string response at step={step}, topic={curr_topic}: {type(response)}")

        if step == 'init_contextual_personal_history':
            # print(utils.Colors.OKGREEN, "Response:", utils.Colors.ENDC, response)
            text_before_json = re.split(r'```json', response)[0].strip()
            # print(utils.Colors.OKGREEN, "Text Before JSON:", utils.Colors.ENDC, text_before_json)
            utils.append_json_to_file(text_before_json, output_file_path, curr_data_name="Topic-Specific Hobbies", parse_json=False)
            try:
                try:
                    json_part = re.split(r'```json', response)[1].strip()
                except Exception:
                    json_part = response
                # print(utils.Colors.OKGREEN, "JSON Part before repair:", utils.Colors.ENDC, json_part)
                json_part = repair_json('[{'+json_part+'}]')
                json_part = utils.filter_valid_dates(json_part)
                utils.append_json_to_file(json_part, output_file_path, curr_data_name=data_name, parse_json=False)
                normalized_history[step] = json_part
                # print(utils.Colors.OKGREEN, "JSON Part after repair:", utils.Colors.ENDC, json_part)
                last_timestamps.append(utils.extract_last_timestamp(json_part))
            except Exception as e:
                preview = (response[:1200] + "...") if isinstance(response, str) and len(response) > 1200 else response
                raise RuntimeError(
                    f"Failed parsing step={step} for topic={curr_topic}. "
                    f"Response preview:\n{preview}\n"
                    f"Original error: {repr(e)}"
                ) from e
        else:
            try:
                response = repair_json('[{'+response+'}]')
                # print('step', step, 'response', type(response), response)
                response = utils.filter_valid_dates(response)
                # print('filtered response', response)
                utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)
                normalized_history[step] = response
                last_timestamps.append(utils.extract_last_timestamp(response))
            except Exception as e:
                preview = (str(response)[:1200] + "...") if len(str(response)) > 1200 else str(response)
                raise RuntimeError(
                    f"Failed parsing step={step} for topic={curr_topic}. "
                    f"Response preview:\n{preview}\n"
                    f"Original error: {repr(e)}"
                ) from e

    # Populate personal history into conversation
    steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
    data_names = CONVERSATION_SECTION_NAMES
    # steps = ['init_conversation']
    # data_names = ['Init Conversation']

    last_timestamps = utils.merge_timestamps(last_timestamps)
    sensitive_info_pool = build_sensitive_info_pool(expanded_persona, getattr(LLM, "pii_profile", None), curr_topic)
    utils.append_json_to_file(sensitive_info_pool, output_file_path, curr_data_name='Sensitive Info Pool', parse_json=False)

    raw_context_histories = [
        normalized_history.get('init_contextual_personal_history', {}),
        normalized_history.get('first_expand_contextual_personal_history', {}),
        normalized_history.get('second_expand_contextual_personal_history', {}),
        normalized_history.get('third_expand_contextual_personal_history', {}),
    ]
    period_keys = ["init", "week", "month", "year"]
    event_histories = [
        build_event_history(raw_context_histories[0], curr_topic, "init", sensitive_info_pool),
        build_event_history(raw_context_histories[1], curr_topic, "week", sensitive_info_pool),
        build_event_history(raw_context_histories[2], curr_topic, "month", sensitive_info_pool),
        build_event_history(raw_context_histories[3], curr_topic, "year", sensitive_info_pool),
    ]
    event_history_names = EVENT_HISTORY_SECTION_NAMES
    for event_name, hist in zip(event_history_names, event_histories):
        utils.append_json_to_file(hist, output_file_path, curr_data_name=event_name, parse_json=False)

    interaction_dates = [
        select_interaction_dates(LLM, event_histories[0], curr_topic, "init", verbose=args['inference']['verbose']),
        select_interaction_dates(LLM, event_histories[1], curr_topic, "week", verbose=args['inference']['verbose']),
        select_interaction_dates(LLM, event_histories[2], curr_topic, "month", verbose=args['inference']['verbose']),
        select_interaction_dates(LLM, event_histories[3], curr_topic, "year", verbose=args['inference']['verbose']),
    ]
    interaction_date_names = INTERACTION_SOURCE_SECTION_NAMES
    for data_name, dates in zip(interaction_date_names, interaction_dates):
        utils.append_json_to_file(dates, output_file_path, curr_data_name=data_name, parse_json=False)

    interaction_histories = [
        build_interaction_history(LLM, event_histories[0], curr_topic, "init", sensitive_info_pool, interaction_dates[0], expanded_persona, normalized_history.get('init_general_personal_history', {}), verbose=args['inference']['verbose']),
        build_interaction_history(LLM, event_histories[1], curr_topic, "week", sensitive_info_pool, interaction_dates[1], expanded_persona, normalized_history.get('first_expand_general_personal_history', {}), verbose=args['inference']['verbose']),
        build_interaction_history(LLM, event_histories[2], curr_topic, "month", sensitive_info_pool, interaction_dates[2], expanded_persona, normalized_history.get('second_expand_general_personal_history', {}), verbose=args['inference']['verbose']),
        build_interaction_history(LLM, event_histories[3], curr_topic, "year", sensitive_info_pool, interaction_dates[3], expanded_persona, normalized_history.get('third_expand_general_personal_history', {}), verbose=args['inference']['verbose']),
    ]
    interaction_history_names = INTERACTION_HISTORY_SECTION_NAMES
    for hist_name, hist in zip(interaction_history_names, interaction_histories):
        utils.append_json_to_file(hist, output_file_path, curr_data_name=hist_name, parse_json=False)

    conversation_histories = [
        build_conversation_history(event_histories[0], interaction_histories[0]),
        build_conversation_history(event_histories[1], interaction_histories[1]),
        build_conversation_history(event_histories[2], interaction_histories[2]),
        build_conversation_history(event_histories[3], interaction_histories[3]),
    ]
    conversation_history_names = CONVERSATION_HISTORY_SECTION_NAMES
    for hist_name, hist in zip(conversation_history_names, conversation_histories):
        utils.append_json_to_file(hist, output_file_path, curr_data_name=hist_name, parse_json=False)

    # Conversation generation should expand every conversation-history item.
    LLM.init_personal_history = json.dumps(conversation_histories[0], ensure_ascii=False, indent=2)
    LLM.first_expand_personal_history = json.dumps(conversation_histories[1], ensure_ascii=False, indent=2)
    LLM.second_expand_personal_history = json.dumps(conversation_histories[2], ensure_ascii=False, indent=2)
    LLM.third_expand_personal_history = json.dumps(conversation_histories[3], ensure_ascii=False, indent=2)

    for conv_idx, (step, data_name) in enumerate(zip(steps, data_names)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        try:
            maybe_write_conversation_log(output_file_path, args, data_name, "expected_history", conversation_histories[conv_idx])
            response = LLM.query_llm(
                step=step,
                topic=curr_topic,
                idx_topic=idx_topic,
                start_time=start_time,
                verbose=args['inference']['verbose'],
                sensitive_info_pool=sensitive_info_pool,
            )
            maybe_write_conversation_log(output_file_path, args, data_name, "raw", response)
            reflected = LLM.query_llm(
                step='reflect_' + step,
                topic=curr_topic,
                data=response,
                action=1,
                verbose=args['inference']['verbose'],
            )
            maybe_write_conversation_log(output_file_path, args, data_name, "reflect_round1", reflected)
            response = LLM.query_llm(step='reflect_' + step, topic=curr_topic, action=2, verbose=args['inference']['verbose'])
            maybe_write_conversation_log(output_file_path, args, data_name, "reflect_round2", response)
            expanded_conversation = parse_conversation_sections(LLM, response, curr_topic, last_timestamps[conv_idx], verbose=args['inference']['verbose'])
            maybe_write_conversation_log(output_file_path, args, data_name, "parsed", expanded_conversation)
            expanded_conversation = _dedupe_side_note_blocks(expanded_conversation)
            maybe_write_conversation_log(output_file_path, args, data_name, "deduped", expanded_conversation)
            _assert_conversation_aligned(expanded_conversation, conversation_histories[conv_idx], data_name)
            utils.append_json_to_file(expanded_conversation, output_file_path, curr_data_name=data_name, parse_json=False, parse_list=False)
        except Exception as e:
            maybe_write_conversation_log(output_file_path, args, data_name, "error", repr(e))
            raise


def prepare_irrelevant_contexts(LLM, args):
    random_questions_dir = args['datasets']['random_questions_dir']
    question_files = sorted(glob.glob(os.path.join(random_questions_dir, "*.txt")))
    if len(question_files) == 0:
        raise FileNotFoundError(f"No .txt files found under {random_questions_dir}")

    all_random_questions = []
    for question_file in question_files:
        with open(question_file, 'r') as file:
            all_random_questions.extend([line.strip() for line in file if line.strip()])

    # Deduplicate while preserving order.
    all_random_questions = list(dict.fromkeys(all_random_questions))
    print(f'Loaded {len(all_random_questions)} irrelevant questions from {len(question_files)} files under {random_questions_dir}.')

    output_file_path = args['datasets']['random_contexts_file']
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    restart_irrelevant = args['inference'].get('restart_irrelevant', False)
    existing_entries = []
    completed_map = {}
    if restart_irrelevant:
        print(f'restart_irrelevant=True, rebuilding {output_file_path} from scratch.')
    elif os.path.exists(output_file_path):
        try:
            with open(output_file_path, "r", encoding="utf-8") as file:
                existing_entries = json.load(file)
            if isinstance(existing_entries, list):
                for item in existing_entries:
                    if isinstance(item, dict) and len(item) == 1:
                        key = next(iter(item.keys()))
                        if str(key).isdigit():
                            completed_map[int(key)] = item[str(key)]
        except (json.JSONDecodeError, OSError):
            completed_map = {}

    if len(completed_map) > 0:
        print(f'Resuming from checkpoint: {len(completed_map)} entries already exist in {output_file_path}.')

    def persist_checkpoint():
        ordered_entries = [{str(idx): completed_map[idx]} for idx in sorted(completed_map.keys())]
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(ordered_entries, file, indent=4)

    for index, question in enumerate(tqdm(all_random_questions, disable=TQDM_DISABLE)):
        if index in completed_map:
            continue

        LLM.create_a_thread(step='irrelevant')
        try:
            model_answer = LLM.query_llm(step='random_question', data=question, verbose=args['inference']['verbose'])
            follow_up_question = LLM.query_llm(step='random_question_follow_up', verbose=args['inference']['verbose'])
            follow_up_answer = LLM.query_llm(step='random_question_follow_up_response', data=follow_up_question, verbose=args['inference']['verbose'])
        except Exception:
            LLM.delete_a_thread(step='irrelevant')
            persist_checkpoint()
            raise

        new_entry = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": model_answer},
            {"role": "user", "content": follow_up_question},
            {"role": "assistant", "content": follow_up_answer}
        ]

        LLM.delete_a_thread(step='irrelevant')
        completed_map[index] = new_entry
        if (index + 1) % 50 == 0:
            persist_checkpoint()

    persist_checkpoint()


def prepare_data(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()

    if args['datasets']['topics'] == ['irrelevant']:
        LLM = QueryLLM(args)
        prepare_irrelevant_contexts(LLM, args)
    else:
        # Generate conversational data relevant to the topic and the persona
        all_errored_data_paths = {}
        persona_indices = range(int(args['inference']['start_persona_idx']), int(args['inference']['num_personas']))

        def process_persona(idx_persona):
            errored = {}
            regenerate_conversation_only = bool(args['inference'].get('regenerate_conversation_only', False))
            LLM = QueryLLM(args)
            if regenerate_conversation_only:
                persona = expanded_persona = None
                init_general_personal_history = general_personal_history_next_week = None
                general_personal_history_next_month = general_personal_history_next_year = None
                parsed_pii = None
                start_time = None
            else:
                persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, \
                    general_personal_history_next_month, general_personal_history_next_year = prepare_persona(LLM, idx_persona, all_personas, args)
                parsed_pii = QueryLLM.parse_synthetic_pii_from_persona_text(expanded_persona)
                if parsed_pii:
                    LLM.pii_profile = parsed_pii

            # Clean up the names of topics
            if args['datasets']['topics'] == ['all']:
                all_topics = utils.get_all_topic_names()
            else:
                all_topics = [ctx.strip() for ctx in args['datasets']['topics']]

            # Since we assign a consecutive time frame for all topics, we randomly permute topics to ensure generalization
            if len(all_topics) > 1:
                random.shuffle(all_topics)

                # Ensure "writing" or "email" is not the first topic
                restricted_topics = {"writing", "email"}
                if all_topics[0] in restricted_topics:
                    for i in range(1, len(all_topics)):
                        if all_topics[i] not in restricted_topics:
                            all_topics[0], all_topics[i] = all_topics[i], all_topics[0]
                            break

            # Loop through each topic in the list
            for idx_topic, curr_topic in tqdm(enumerate(all_topics), disable=TQDM_DISABLE):
                if curr_topic == '' or curr_topic is None:
                    continue
                source_dir, all_source_files = prepare_topics(idx_topic, all_topics, curr_topic, args)

                # Set a consecutive time frame for different topics for each persona, while all samples below are independent
                if idx_topic > 0 and start_time is not None:
                    start_time = utils.pick_a_random_time_within_a_year(start_time)

                for idx_sample in range(int(args['inference']['start_sample_idx']), int(args['inference']['num_samples_per_topic'])):
                    output_file_path = os.path.join(args['inference']['output_dir'],
                                                    os.path.join(f'{curr_topic}', f'{args["inference"]["output_file_name"]}_{curr_topic}_persona{idx_persona}_sample{idx_sample}.json'))
                    if (
                        args['inference'].get('skip_existing', False)
                        and not args['inference'].get('regenerate_conversation_only', False)
                        and is_output_complete(output_file_path, curr_topic)
                    ):
                        print(f'{utils.Colors.WARNING}Skipping existing complete file: {output_file_path}{utils.Colors.ENDC}')
                        continue

                    # Load a random source data to the LLM as a background memory about the topic
                    source_data = utils.load_one_source_data(source_dir, all_source_files, curr_topic) if all_source_files is not None else None
                    max_retries = int(args['inference'].get('max_retries', 0))
                    retry_backoff = float(args['inference'].get('retry_backoff', 2.0))
                    success = False
                    for attempt in range(max_retries + 1):
                        if attempt > 0 and os.path.exists(output_file_path) and not regenerate_conversation_only:
                            os.remove(output_file_path)
                        try:
                            LLM = QueryLLM(args)
                            if parsed_pii:
                                LLM.pii_profile = parsed_pii
                            if regenerate_conversation_only and os.path.exists(output_file_path):
                                with open(output_file_path, "r", encoding="utf-8") as f:
                                    existing_data = json.load(f)
                                missing_prereqs = get_missing_conversation_prereq_sections(existing_data, curr_topic)
                                if missing_prereqs:
                                    raise RuntimeError(
                                        f"cannot regenerate conversations only; missing prerequisite sections {missing_prereqs}"
                                    )
                                print(
                                    f'{utils.Colors.OKGREEN}Reusing existing history and rewriting only conversations: '
                                    f'{output_file_path}{utils.Colors.ENDC}'
                                )
                                LLM.create_a_thread(step='conversation')
                                rewrite_conversations_from_existing(
                                    LLM,
                                    existing_data,
                                    curr_topic,
                                    start_time,
                                    output_file_path,
                                    args,
                                )
                                with open(output_file_path, "r", encoding="utf-8") as f:
                                    generated = json.load(f)
                                missing_sections = get_missing_output_sections(generated, curr_topic)
                                if missing_sections:
                                    raise RuntimeError(
                                        f"incomplete output: missing sections {missing_sections} for {output_file_path}"
                                    )
                                success = True
                                break
                            utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                            utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                            if parsed_pii:
                                utils.append_json_to_file(parsed_pii, output_file_path, curr_data_name='Persona PII', parse_json=False)
                            utils.append_json_to_file(curr_topic, output_file_path, curr_data_name='Topic', parse_json=False)
                            print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path} (attempt {attempt + 1}){utils.Colors.ENDC}')

                            if curr_topic == 'writing' or curr_topic == 'email':
                                """
                                Besides other topics, we introduce creative writing and email writing when evaluating the LLM's ability to generate persona-aligned new contents.
                                It is meaningful as a special case since it is (1) practically useful (2) need to translate writing samples into conversations (3) does not involve personal historical events as in other topics.
                                """
                                LLM.create_a_thread(step='writing')
                                prepare_data_on_writing_topic(LLM, curr_topic, persona, source_data, output_file_path, args)
                            else:
                                LLM.create_a_thread(step='conversation')
                                prepare_data_on_other_topics(
                                    LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                    init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year, args
                                )
                            try:
                                with open(output_file_path, "r", encoding="utf-8") as f:
                                    generated = json.load(f)
                            except Exception as e:
                                raise RuntimeError(f"incomplete output: unreadable json after generation ({e})")

                            missing_sections = get_missing_output_sections(generated, curr_topic)
                            if missing_sections:
                                raise RuntimeError(
                                    f"incomplete output: missing sections {missing_sections} for {output_file_path}"
                                )
                            success = True
                            break
                        except Exception as e:
                            if attempt < max_retries and is_retryable_error(e):
                                sleep_seconds = retry_backoff * (2 ** attempt)
                                print(f'{utils.Colors.WARNING}Retryable error at {output_file_path}: {e}. Retrying in {sleep_seconds:.1f}s...{utils.Colors.ENDC}')
                                time.sleep(sleep_seconds)
                                continue
                            print(f'{utils.Colors.FAIL}Error at generating file {output_file_path}: {repr(e)}{utils.Colors.ENDC}')
                            traceback.print_exc()
                            errored[output_file_path] = str(e)
                            break
                    if not success and output_file_path not in errored:
                        errored[output_file_path] = "Unknown failure"
            return errored

        workers = int(args['inference'].get('workers', 1))
        if workers <= 1:
            for idx_persona in tqdm(persona_indices, disable=TQDM_DISABLE):
                curr_errors = process_persona(idx_persona)
                all_errored_data_paths.update(curr_errors)
        else:
            print(f'{utils.Colors.OKBLUE}Running with {workers} workers across persona shards.{utils.Colors.ENDC}')
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_persona, idx): idx for idx in persona_indices}
                for fut in tqdm(as_completed(futures), total=len(futures), disable=TQDM_DISABLE):
                    curr_errors = fut.result()
                    all_errored_data_paths.update(curr_errors)

        if len(all_errored_data_paths) > 0:
            print(f'{utils.Colors.FAIL}All errored data paths: {utils.Colors.ENDC}')
            for key, value in all_errored_data_paths.items():
                print(key)
        else:
            print(f'{utils.Colors.OKGREEN}All data are successfully generated.{utils.Colors.ENDC}')


if __name__ == "__main__":
    print("Python", sys.version, 'Torch', torch.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    torch.manual_seed(0)
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if world_size > 1:
        print(f'{utils.Colors.WARNING}Detected {world_size} GPUs. Defaulting to a single GPU run on cuda:0.{utils.Colors.ENDC}')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--api_mode', type=str, default='auto', choices=['auto', 'assistants', 'responses', 'chat'],
                        help='OpenAI API mode for generation: auto, assistants, responses, or chat')
    parser.add_argument('--topics', type=str, default="therapy", nargs="+",
                            help='Set conversation topics. Choose from therapy, legalConsultation, datingConsultation, foodRecommendation, onlineShopping, studyConsultation, '
                                 'travelPlanning, movieRecommendation, songRecommendation, homeDecoration, financialConsultation, healthConsultation, writing.'
                                 'or all to select all existing topics under ./data/output/. '
                                 'If you want to select multiple topics manually, separate the names by space, e.g. --topics therapy legal'
                                 'Choose "irrelevant" if you want to generate data irrelevant to the topic to fill in long conversation context')  # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument('--n_persona', type=int, default=1, help='Set number of personas to generate')
    parser.add_argument('--n_samples', type=int, default=1, help='Set number of samples per topic to generate')
    parser.add_argument('--s_persona', type=int, default=0, help='Set the starting idx of personas to generate')
    parser.add_argument('--s_samples', type=int, default=0, help='Set the starting idx of samples per topic to generate')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument('--output_dir', type=str, default='data/output/', help='Set the path to the output directory')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--restart_irrelevant', dest='restart_irrelevant', action='store_true',
                        help='When --topics irrelevant, ignore existing irrelevant_contexts checkpoint and regenerate from scratch')
    parser.add_argument('--skip_existing', dest='skip_existing', action='store_true',
                        help='Skip already completed output files for non-irrelevant topics')
    parser.add_argument('--workers', type=int, default=20,
                        help='Number of worker threads (persona-level parallelism)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Max retries for retryable API/network errors')
    parser.add_argument('--retry_backoff', type=float, default=2.0,
                        help='Base seconds for exponential backoff between retries')
    parser.add_argument('--regenerate_conversation_only', dest='regenerate_conversation_only', action='store_true',
                        help='Reuse existing generated content and rewrite only the conversation sections')
    parser.add_argument('--force_regen_persona', dest='force_regen_persona', action='store_true',
                        help='Ignore existing persona cache and regenerate Original/Expanded Persona')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['models']['api_mode'] = cmd_args.api_mode if cmd_args.api_mode is not None else args['models'].get('api_mode', 'auto')
    args['datasets']['topics'] = cmd_args.topics if cmd_args.topics is not None else args['datasets']['topics']
    args['inference']['num_personas'] = cmd_args.n_persona if cmd_args.n_persona is not None else args['inference']['num_personas']
    args['inference']['num_samples_per_topic'] = cmd_args.n_samples if cmd_args.n_samples is not None else args['inference']['num_samples_per_topic']
    args['inference']['start_persona_idx'] = cmd_args.s_persona if cmd_args.s_persona is not None else args['inference']['start_persona_idx']
    args['inference']['start_sample_idx'] = cmd_args.s_samples if cmd_args.s_samples is not None else args['inference']['start_sample_idx']
    args['inference']['output_dir'] = cmd_args.output_dir if cmd_args.output_dir is not None else args['inference']['output_dir']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['inference']['restart_irrelevant'] = cmd_args.restart_irrelevant
    args['inference']['skip_existing'] = cmd_args.skip_existing
    args['inference']['workers'] = cmd_args.workers
    args['inference']['max_retries'] = cmd_args.max_retries
    args['inference']['retry_backoff'] = cmd_args.retry_backoff
    args['inference']['regenerate_conversation_only'] = cmd_args.regenerate_conversation_only
    args['inference']['force_regen_persona'] = cmd_args.force_regen_persona

    # Start inference
    print(args)
    if cmd_args.clean:
        user_input = input("The 'clean' flag is set. Do you really want clean up all existing data under ./data/output/? (y/n): ").strip().lower()
        if user_input == 'y':
            utils.clean_up_subdirectories()
        else:
            print("Skipping cleanup.")

    prepare_data(args)
