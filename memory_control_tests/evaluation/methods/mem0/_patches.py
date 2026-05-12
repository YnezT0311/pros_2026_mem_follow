"""Defensive patches we apply to a self-hosted mem0 `Memory` instance.

The mem0 paper's evaluation runs against the managed `MemoryClient`, where the
mem0 service handles fact extraction, UPDATE_MEMORY validation, and history
storage. This benchmark runs mem0 self-hosted with local Qdrant and a smaller
OpenRouter model, which means we have to reproduce the validation that the
managed service performs internally:

  * `MEM0_STRICT_UPDATE_MEMORY_PROMPT` — prepended to UPDATE_MEMORY so the LLM
    is told not to invent IDs (smaller models hallucinate IDs without it).
  * `install_update_action_guard` — server-side post-validation: drops
    UPDATE/DELETE actions that reference IDs not present in the prompt's
    current-memory block, and rewrites them as ADD when they carry text.
  * `disable_history_writes` — silence the local sqlite history writer so
    parallel workers do not contend on the lock file.

Also keeps the LLM-trace and fact-extraction debug helpers so eval outputs can
be compared across unified `mem_evals.py` runs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


MEM0_STRICT_UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

When deciding memory actions, follow these rules strictly:
- Only use existing IDs from the provided current memory block for UPDATE, DELETE, and NONE.
- Never invent or hallucinate an existing ID for UPDATE, DELETE, or NONE.
- If a fact seems related to an old memory but you are not fully sure which existing ID matches it, use ADD instead of UPDATE.
- If you want to delete something but there is no clearly matching existing ID in the provided current memory block, do not delete it.
- For ADD actions, generate a fresh new ID that does not overlap with the provided current-memory IDs.
- Return only valid JSON in the requested schema.
"""


MEM0_LOCOMO_PROJECT_CUSTOM_INSTRUCTIONS = """Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


# ---------- update-memory response sanitizer ----------

def _extract_first_fenced_block(text: str) -> str:
    match = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_valid_update_ids_from_prompt(prompt: str) -> List[str]:
    marker = "Below is the current content of my memory"
    marker_index = prompt.find(marker)
    if marker_index == -1:
        return []
    prompt_tail = prompt[marker_index:]
    memory_block = _extract_first_fenced_block(prompt_tail)
    if not memory_block:
        return []
    ids = re.findall(r"""['"]id['"]\s*:\s*['"]([^'"]+)['"]""", memory_block)
    deduped: List[str] = []
    for item in ids:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _next_memory_id(existing_ids: List[str], reserved_ids: List[str]) -> str:
    numeric_ids: List[int] = []
    for item in [*existing_ids, *reserved_ids]:
        try:
            numeric_ids.append(int(item))
        except Exception:
            continue
    if numeric_ids:
        return str(max(numeric_ids) + 1)
    return str(len(existing_ids) + len(reserved_ids))


def sanitize_update_memory_response_text(prompt: str, response_text: str) -> str:
    if "The new retrieved facts are mentioned in the triple backticks" not in prompt:
        return response_text
    if not isinstance(response_text, str) or not response_text.strip():
        return response_text

    cleaned = response_text.strip()
    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(cleaned, strict=False)
    except Exception:
        try:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                parsed = json.loads(cleaned[start_idx : end_idx + 1], strict=False)
        except Exception:
            parsed = None
    if not isinstance(parsed, dict):
        return response_text

    memory_actions = parsed.get("memory")
    if not isinstance(memory_actions, list):
        return response_text

    valid_ids = _extract_valid_update_ids_from_prompt(prompt)
    valid_id_set = set(valid_ids)
    reserved_new_ids: List[str] = []
    sanitized_actions: List[Dict[str, Any]] = []

    for action in memory_actions:
        if not isinstance(action, dict):
            continue
        event = str(action.get("event", "")).upper().strip()
        action_id = str(action.get("id", "")).strip()
        action_text = str(action.get("text", "")).strip()
        sanitized = dict(action)

        if event == "ADD":
            if not action_id or action_id in valid_id_set or action_id in reserved_new_ids:
                sanitized["id"] = _next_memory_id(valid_ids, reserved_new_ids)
            reserved_new_ids.append(str(sanitized["id"]))
            sanitized_actions.append(sanitized)
            continue

        if event == "UPDATE":
            if action_id in valid_id_set:
                sanitized_actions.append(sanitized)
                continue
            if action_text:
                sanitized["event"] = "ADD"
                sanitized.pop("old_memory", None)
                sanitized["id"] = _next_memory_id(valid_ids, reserved_new_ids)
                reserved_new_ids.append(str(sanitized["id"]))
                sanitized_actions.append(sanitized)
            continue

        if event == "DELETE":
            if action_id in valid_id_set:
                sanitized_actions.append(sanitized)
            continue

        if event == "NONE":
            if action_id in valid_id_set:
                sanitized_actions.append(sanitized)
            continue

        sanitized_actions.append(sanitized)

    parsed["memory"] = sanitized_actions
    return json.dumps(parsed, ensure_ascii=False)


def install_update_action_guard(memory: Any) -> None:
    """Wrap `memory.llm.generate_response` to validate UPDATE_MEMORY output."""
    if getattr(memory.llm, "_memoryctrl_update_guard_applied", False):
        return
    original_generate_response = memory.llm.generate_response

    def guarded_generate_response(*args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        response = original_generate_response(*args, **kwargs)
        if not isinstance(messages, list) or len(messages) != 1:
            return response
        message = messages[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            return response
        prompt = message.get("content", "")
        if not isinstance(prompt, str):
            return response
        return sanitize_update_memory_response_text(prompt, response)

    memory.llm.generate_response = guarded_generate_response
    memory.llm._memoryctrl_update_guard_applied = True


def disable_history_writes(memory: Any) -> None:
    """Silence the sqlite history table — it causes worker contention."""
    if not hasattr(memory, "db"):
        return

    def _noop_add_history(*args: Any, **kwargs: Any) -> None:
        return None

    memory.db.add_history = _noop_add_history


# ---------- preload tracing ----------

def run_add_with_llm_trace(
    memory: Any,
    clean_messages: List[Dict[str, str]],
    *,
    user_id: str,
    run_id: str,
    usage_log: Dict[str, int] | None = None,
) -> tuple[List[Any], List[Dict[str, Any]]]:
    llm_calls: List[Dict[str, Any]] = []
    original_generate = memory.llm.generate_response

    # Also hook the underlying chat.completions.create so we can record
    # `response.usage` (prompt_tokens / completion_tokens). mem0's
    # generate_response parses to text and discards usage, so without this
    # bypass the adapter couldn't report internal-LLM cost.
    underlying_client = getattr(memory.llm, "client", None)
    original_create = None
    if usage_log is not None and underlying_client is not None:
        try:
            original_create = underlying_client.chat.completions.create

            def counting_create(*args: Any, **kwargs: Any):
                resp = original_create(*args, **kwargs)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                    ct = int(getattr(usage, "completion_tokens", 0) or 0)
                    usage_log["calls"] = usage_log.get("calls", 0) + 1
                    usage_log["prompt_tokens"] = usage_log.get("prompt_tokens", 0) + pt
                    usage_log["completion_tokens"] = usage_log.get("completion_tokens", 0) + ct
                    usage_log["total_tokens"] = usage_log.get("total_tokens", 0) + pt + ct
                return resp

            underlying_client.chat.completions.create = counting_create
        except Exception:
            original_create = None  # bail; preload still works without counts

    def traced_generate_response(*args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        call_record: Dict[str, Any] = {
            "messages": messages,
            "response_format": kwargs.get("response_format"),
            "response": None,
            "error": None,
        }
        try:
            response = original_generate(*args, **kwargs)
            call_record["response"] = response
            return response
        except Exception as exc:
            call_record["error"] = repr(exc)
            raise
        finally:
            llm_calls.append(call_record)

    memory.llm.generate_response = traced_generate_response
    try:
        add_results: List[Any] = [
            {
                "batch_index": 1,
                "batch_messages": clean_messages,
                "result": memory.add(clean_messages, user_id=user_id, run_id=run_id),
            }
        ]
    finally:
        memory.llm.generate_response = original_generate
        if original_create is not None and underlying_client is not None:
            underlying_client.chat.completions.create = original_create
    return add_results, llm_calls


def normalize_items(raw_items: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_items, list):
        return [item for item in raw_items if isinstance(item, dict)]
    if isinstance(raw_items, dict):
        nested = raw_items.get("results")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def snapshot_store(memory: Any, *, user_id: str, run_id: str, limit: int = 200) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "raw": None,
        "normalized_items": [],
        "count": 0,
    }
    raw = memory.get_all(user_id=user_id, run_id=run_id, limit=limit)
    snapshot["raw"] = raw
    snapshot["normalized_items"] = normalize_items(raw)
    snapshot["count"] = len(snapshot["normalized_items"])
    return snapshot


def format_memories(search_result: Any) -> str:
    results = normalize_items(search_result)
    if not results:
        return "No relevant memories were retrieved."

    lines: List[str] = []
    for idx, item in enumerate(results, start=1):
        memory = item.get("memory", "") if isinstance(item, dict) else ""
        score = item.get("score") if isinstance(item, dict) else None
        if score is None:
            lines.append(f"{idx}. {memory}")
        else:
            lines.append(f"{idx}. {memory} (score={score:.4f})")
    return "\n".join(lines)
