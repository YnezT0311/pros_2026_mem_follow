#!/usr/bin/env python3
"""
evaluate_claude_web.py

Claude Web evaluator scaffold for MemoryCtrl data.

Current status:
  - Supports manual login with a persistent browser profile, mirroring the
    ChatGPT evaluator flow.
  - Reuses the existing MemoryCtrl session-planning / output-artifact logic.
  - Leaves Claude-specific selectors and memory-management UI hooks isolated in
    one place so they can be completed once we inspect screenshots / click logs.

Intended workflow:
  1. Run with --login and complete Claude login manually in the browser.
  2. Share screenshots / click-recorder output for the composer, new-chat,
     delete-chat, and added-memory controls.
  3. Fill in provider-specific selectors and helper implementations.
"""

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from patchright.async_api import BrowserContext, Page, async_playwright


CLAUDE_URL = "https://claude.ai/"
DEFAULT_TIMING_PROFILE = str(Path(__file__).with_name("human_timing.json"))

# These are intentionally broad starter selectors. We'll tighten them after we
# inspect the real Claude Web DOM / screenshots from your account.
SEL_INPUT_CANDIDATES = [
    'div[contenteditable="true"]',
    'textarea',
]
SEL_ASSISTANT_MESSAGE_CANDIDATES = [
    '[data-testid="assistant-message"]',
    '[data-is-streaming]',
    'div[data-message-author="assistant"]',
    'div[data-test-render-count]',
]
SEL_SEND_BUTTON_CANDIDATES = [
    'button[aria-label*="Send"]',
    'button[aria-label*="send"]',
    'button[data-testid*="send"]',
]
SEL_STOP_BUTTON_CANDIDATES = [
    'button[aria-label*="Stop"]',
    'button[aria-label*="stop"]',
    'button[data-testid*="stop"]',
]
SEL_DELETE_CHAT_TRIGGER = '[data-testid="delete-chat-trigger"]'
SEL_DELETE_CONFIRM = '[data-testid="delete-modal-confirm"]'
SEL_CHAT_MENU_TRIGGER = 'button[aria-label^="More options for "]'
SEL_CHAT_ROW = 'a[data-dd-action-name="sidebar-chat-item"]'
SEL_CHAT_ROW_MENU_TRIGGER = 'a[data-dd-action-name="sidebar-chat-item"] + div button[aria-label^="More options for "]'
SEL_ADDED_MEMORY_BUTTON = 'button[aria-expanded]:has(span:has-text("Added memory"))'
SEL_ADDED_MEMORY_STATUS = 'span[role="status"][aria-live="polite"]'
SEL_ASK_USER_OPTIONS = 'button[id^="ask-user-option-question-"]'
SEL_ASK_USER_OPTION_CANDIDATES = [
    'button[id^="ask-user-option-question-"]',
    'div[role="listbox"] button[role="option"]',
    '[data-ask-user-input-banner="true"] button[role="option"]',
]
SEL_INTERACTIVE_SKIP_CANDIDATES = [
    'button[data-widget-action="skip"]',
    'button:has-text("Skip")',
]
SEL_INTERACTIVE_CHECKBOX_CANDIDATES = [
    'input[type="checkbox"]',
    'button[role="checkbox"]',
    '[data-ask-user-input-banner="true"] input[type="checkbox"]',
]

SEL_RATE_LIMIT_BAND_CANDIDATES = [
    'div[data-alert-band-wrapper="true"]',
    'div[role="status"][aria-live="polite"][aria-atomic="true"]',
]
RATE_LIMIT_TEXT_MARKERS = [
    # Headline phrases (highly specific — vanishingly unlikely in chat content)
    "usage limit reached",
    "message limit reached",
    "rate limit reached",
    "you've hit your session limit",
    "you've hit your daily limit",
    "you've hit your weekly limit",
    "you've hit your usage limit",
    "you've hit your message limit",
    "you've hit your limit for",
    "you've hit the limit",
    "you've reached your message limit",
    "you've reached your usage limit",
    "you've reached your session limit",
    "you've reached your limit",
    # Reset-clause phrases
    "your limit will reset",
    "you'll be able to send messages again",
    "you'll be able to chat again",
]

PERIODS = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]
PERIOD_SHORT = {
    "Conversation Initial Stage": "initial",
    "Conversation Early Stage": "early",
    "Conversation Intermediate Stage": "intermediate",
    "Conversation Late Stage": "late",
}


@dataclass
class TimingProfile:
    typing_chars_per_sec_mean: float = 6.0
    typing_chars_per_sec_std: float = 1.5
    pre_send_pause_mean: float = 1.0
    pre_send_pause_std: float = 0.5
    reading_chars_per_sec_mean: float = 20.0
    reading_chars_per_sec_std: float = 5.0
    post_reading_pause_mean: float = 3.0
    post_reading_pause_std: float = 1.5
    min_turn_delay: float = 3.0
    max_turn_delay: float = 120.0
    n_interactions: int = 0

    @classmethod
    def load(cls, path: str) -> "TimingProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def sample_typing_delay(self, msg_len: int) -> float:
        # In the current Claude web script we mostly use fill()/instant insert,
        # so this value should reflect only the short human pause between
        # finishing input and clicking send, not full typing time.
        pause = random.gauss(self.pre_send_pause_mean, self.pre_send_pause_std)
        return min(0.6, max(0.1, pause))

    def sample_reading_delay(self, response_len: int) -> float:
        rps = max(5.0, random.gauss(self.reading_chars_per_sec_mean, self.reading_chars_per_sec_std))
        thinking = max(0.5, random.gauss(self.post_reading_pause_mean, self.post_reading_pause_std))
        return max(self.min_turn_delay, min(self.max_turn_delay, response_len / rps + thinking))


@dataclass
class McqItem:
    sample_id: str
    qa_family: str
    timestamp: str
    turn_role: str
    identifier_label: str
    user_turn: str
    sensitive_key: str
    sensitive_value: str
    question: str
    choices: dict
    choice_order: list
    choice_to_answer_type: dict
    correct_choice: str


@dataclass
class Phase:
    label: str
    user_turns: list[str]
    mcq_items: list[McqItem] = field(default_factory=list)


@dataclass
class TestSession:
    session_key: str
    phases: list[Phase]


@dataclass
class SessionRunResult:
    records: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    log_lines: list[str]


def _append_text_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()) or "unknown"


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_mcq_items(data_dir: str, topic: str, sample_id: str) -> list[McqItem]:
    items: list[McqItem] = []
    root = Path(data_dir) / "test" / topic

    wr_path = root / "whole_recall" / f"whole_recall_qa_{sample_id}.json"
    if wr_path.exists():
        for item in json.loads(wr_path.read_text(encoding="utf-8")).get("items", []):
            r = item["rendered"]
            items.append(McqItem(
                sample_id=sample_id,
                qa_family="whole_recall",
                timestamp=item["timestamp"],
                turn_role=item.get("turn_role", ""),
                identifier_label=item.get("identifier_label", ""),
                user_turn=item.get("user_turn", ""),
                sensitive_key="",
                sensitive_value="",
                question=r["question"],
                choices=r["choices"],
                choice_order=r["choice_order"],
                choice_to_answer_type=r["choice_to_answer_type"],
                correct_choice=r["remember_correct_choice"],
            ))

    sr_path = root / "slot_recall" / f"slot_recall_qa_{sample_id}.json"
    if sr_path.exists():
        for item in json.loads(sr_path.read_text(encoding="utf-8")).get("items", []):
            for slot in item["rendered"]["items"]:
                items.append(McqItem(
                    sample_id=sample_id,
                    qa_family="slot_recall",
                    timestamp=item["timestamp"],
                    turn_role=item.get("turn_role", ""),
                    identifier_label=item.get("identifier_label", ""),
                    user_turn=item.get("user_turn", ""),
                    sensitive_key=slot["sensitive_key"],
                    sensitive_value=slot["sensitive_value"],
                    question=slot["question"],
                    choices=slot["choices"],
                    choice_order=slot["choice_order"],
                    choice_to_answer_type=slot["choice_to_answer_type"],
                    correct_choice=slot["remember_correct_choice"],
                ))
    return items


def _discover_samples(data_dir: str, topic: str) -> list[str]:
    wr_dir = Path(data_dir) / "test" / topic / "whole_recall"
    prefix = f"whole_recall_qa_{topic}_"
    return sorted(f"{topic}_{p.stem[len(prefix):]}" for p in wr_dir.glob(f"{prefix}*.json"))


def _load_conv(sample_id: str, data_dir: str, world: str) -> tuple[dict, str]:
    topic = sample_id.split("_")[0]
    root = Path(data_dir)
    if world == "baseline":
        path = root / "baseline" / topic / f"conversation_{sample_id}.json"
    else:
        path = root / "test" / topic / world / "transformed_histories" / f"conversation_{sample_id}.{world}.transformed_history.json"
    if not path.exists():
        raise FileNotFoundError(f"Conversation not found for {sample_id} (world={world}):\n  {path}")
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _stage_user_turns(conv: dict, stage: str) -> list[str]:
    return [
        line[5:].strip()
        for line in conv.get(stage, [])
        if isinstance(line, str) and line.startswith("User:")
    ]


def _cumulative_user_turns(conv: dict, up_to_period: str) -> list[str]:
    end_idx = PERIODS.index(up_to_period)
    turns = []
    for period in PERIODS[: end_idx + 1]:
        turns.extend(_stage_user_turns(conv, period))
    return turns


def _detect_forget_stage_per_key(conv: dict, mcq_items: list[McqItem]) -> dict[str, str]:
    forget_kw = (
        "please forget",
        "i want you to delete anything i shared earlier about",
        "please remove from memory anything i shared earlier about",
        "forget anything i told you earlier about",
        "please clear anything i shared about",
        "i do not want you to keep anything i shared about",
    )
    forget_stages: list[str] = []
    for stage in PERIODS:
        for line in conv.get(stage, []):
            if isinstance(line, str) and line.startswith("User:") and any(kw in line.lower() for kw in forget_kw):
                forget_stages.append(stage)
                break

    key_timestamps: list[str] = []
    seen: set[str] = set()
    for item in mcq_items:
        if item.turn_role != "key" or item.timestamp in seen:
            continue
        seen.add(item.timestamp)
        key_timestamps.append(item.timestamp)
    return {ts: stage for ts, stage in zip(key_timestamps, forget_stages)}


def _plan_sessions(sample_id: str, conv: dict, world: str, mcq_items: list[McqItem]) -> list[TestSession]:
    _ = sample_id
    all_mcq = list(mcq_items)

    if world == "baseline":
        items = list(all_mcq)
        random.shuffle(items)
        return [TestSession(
            session_key="baseline_late",
            phases=[Phase(
                label="full history",
                user_turns=_cumulative_user_turns(conv, "Conversation Late Stage"),
                mcq_items=items,
            )],
        )]

    if world == "no_store":
        sessions = []
        for period in [
            "Conversation Early Stage",
            "Conversation Intermediate Stage",
            "Conversation Late Stage",
        ]:
            items = list(all_mcq)
            random.shuffle(items)
            sessions.append(TestSession(
                session_key=f"no_store_{PERIOD_SHORT[period]}",
                phases=[Phase(
                    label=f"up to {PERIOD_SHORT[period]}",
                    user_turns=_cumulative_user_turns(conv, period),
                    mcq_items=items,
                )],
            ))
        return sessions

    if world == "forget":
        forget_stage_map = _detect_forget_stage_per_key(conv, mcq_items)
        if not forget_stage_map:
            print("  WARNING: no forget stages detected — skipping forget session")
            return []

        by_ts: dict[str, list[McqItem]] = {}
        for item in mcq_items:
            by_ts.setdefault(item.timestamp, []).append(item)

        probe_ts_ordered: list[str] = []
        seen_probe_ts: set[str] = set()
        for item in mcq_items:
            if item.turn_role != "probe" or item.timestamp in seen_probe_ts:
                continue
            seen_probe_ts.add(item.timestamp)
            probe_ts_ordered.append(item.timestamp)
        probe_groups = [[i for i in by_ts.get(ts, []) if i.turn_role == "probe"] for ts in probe_ts_ordered]

        phases: list[Phase] = [Phase(
            label="initial",
            user_turns=_stage_user_turns(conv, "Conversation Initial Stage"),
            mcq_items=[],
        )]

        key_ts_ordered = list(forget_stage_map.keys())
        for idx, key_ts in enumerate(key_ts_ordered):
            forget_stage = forget_stage_map[key_ts]
            key_mcqs = [i for i in by_ts.get(key_ts, []) if i.turn_role == "key"]
            probe_mcqs = probe_groups[idx] if idx < len(probe_groups) else []
            phase_items = key_mcqs + probe_mcqs
            random.shuffle(phase_items)
            phases.append(Phase(
                label=f"{PERIOD_SHORT[forget_stage]}_key{idx+1}",
                user_turns=_stage_user_turns(conv, forget_stage),
                mcq_items=phase_items,
            ))
        return [TestSession(session_key="forget_interleaved", phases=phases)]

    raise ValueError(f"Unknown world: {world}")


def _build_mcq_prompt(item: McqItem) -> str:
    options = "\n".join(f"({lbl.lower()}) {item.choices[lbl]}" for lbl in item.choice_order)
    label_str = ", ".join(f"({lbl.lower()})" for lbl in item.choice_order)
    return (
        f"{item.question}\n"
        f"Choose the most appropriate response you would give. "
        f"Answer with just the letter {label_str}.\n\n"
        f"{options}"
    )


def _extract_choice(text: str, choice_order: list[str]) -> str:
    upper = text.upper()
    for lbl in reversed(choice_order):
        if f"({lbl})" in upper:
            return lbl
    for lbl in reversed(choice_order):
        if re.search(rf"\b{lbl}\b", upper):
            return lbl
    return ""


async def _capture_debug_artifacts(page: Page, debug_dir: Path, attempt_index: int, label: str) -> dict[str, str]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"attempt_{attempt_index}_{_slug(label)}"
    screenshot_path = debug_dir / f"{stem}.png"
    html_path = debug_dir / f"{stem}.html"
    meta_path = debug_dir / f"{stem}.json"

    meta: dict[str, Any] = {
        "attempt_index": attempt_index,
        "label": label,
        "timestamp_unix": time.time(),
    }
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
        meta["screenshot_path"] = str(screenshot_path)
    except Exception as e:
        meta["screenshot_error"] = str(e)
    try:
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")
        meta["html_path"] = str(html_path)
    except Exception as e:
        meta["html_error"] = str(e)
    try:
        meta["page_url"] = page.url
    except Exception:
        pass
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in meta.items() if isinstance(v, str)}


def _session_is_completed(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed"


def _print_summary(output_path: Path) -> None:
    if not output_path.exists():
        return
    records = [r for r in _load_jsonl(str(output_path)) if r.get("qa_family")]
    if not records:
        return
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    answered = [r for r in records if not r.get("error") and r.get("predicted_choice")]
    correct = sum(1 for r in answered if r.get("predicted_choice") == r.get("correct_choice"))
    print("\n=== Summary ===")
    print(f"Total: {total}  Errors: {errors}  Answered: {len(answered)}")
    if answered:
        print(f"Overall correct: {correct}/{len(answered)} ({100*correct/len(answered):.1f}%)")


def _claude_results_root(output_path: Path) -> Path:
    return output_path.parent / "claude_web_results"


def _claude_persona_output_path(output_path: Path, topic: str, world: str, sample_id: str) -> Path:
    return _claude_results_root(output_path) / _slug(topic) / _slug(sample_id) / f"test_type_{_slug(world)}" / "results.jsonl"


def _claude_session_artifact_paths(output_path: Path, topic: str, world: str, sample_id: str, session_key: str) -> dict[str, Path]:
    session_dir = _claude_results_root(output_path) / _slug(topic) / _slug(sample_id) / f"test_type_{_slug(world)}" / f"session_{_slug(session_key)}"
    return {
        "dir": session_dir,
        "result": session_dir / "session_result.json",
        "trace": session_dir / "session_trace.json",
        "trace_jsonl": session_dir / "session_trace.jsonl",
        "log": session_dir / "session_log.txt",
        "debug_dir": session_dir / "debug",
    }


def _write_claude_session_artifacts(
    output_path: Path,
    topic: str,
    world: str,
    sample_id: str,
    session: TestSession,
    conv_source: str,
    session_result: SessionRunResult,
    status: str,
    error: str = "",
) -> None:
    import json

    artifacts = _claude_session_artifact_paths(output_path, topic, world, sample_id, session.session_key)
    artifacts["dir"].mkdir(parents=True, exist_ok=True)
    answered = [r for r in session_result.records if not r.get("error") and r.get("predicted_choice")]
    correct = sum(1 for r in answered if r.get("predicted_choice") == r.get("correct_choice"))
    result_payload = {
        "topic": topic,
        "world": world,
        "test_type": world,
        "sample_id": sample_id,
        "session_key": session.session_key,
        "status": status,
        "error": error,
        "conv_source": conv_source,
        "num_phases": len(session.phases),
        "num_records": len(session_result.records),
        "num_answered": len(answered),
        "num_correct": correct,
        "accuracy": (correct / len(answered)) if answered else 0.0,
        "phase_labels": [p.label for p in session.phases],
        "records_path": str(artifacts["trace"]),
        "trace_jsonl_path": str(artifacts["trace_jsonl"]),
        "log_path": str(artifacts["log"]),
        "records": session_result.records,
    }
    artifacts["result"].write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["trace"].write_text(json.dumps(session_result.trace, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["log"].write_text("\n".join(session_result.log_lines) + "\n", encoding="utf-8")


async def _wait_for_any_selector(page: Page, selectors: list[str], timeout: int) -> str:
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        for sel in selectors:
            try:
                if await page.locator(sel).first.is_visible(timeout=300):
                    return sel
            except Exception:
                pass
        await page.wait_for_timeout(300)
    raise TimeoutError(f"None of the candidate selectors became visible: {selectors}")


async def _wait_for_claude_ready(page: Page, timeout: int = 300_000) -> str:
    return await _wait_for_any_selector(page, SEL_INPUT_CANDIDATES, timeout)


async def _new_chat(page: Page, on_interstitial=None) -> None:
    # Placeholder implementation: navigating to Claude home/new chat is the
    # least assumptions-heavy path before we lock down selectors.
    await page.goto(CLAUDE_URL)
    await page.wait_for_timeout(2000)
    await _handle_claude_interstitials(page, on_interstitial=on_interstitial)


async def _handle_claude_interstitials(page: Page, on_interstitial=None) -> bool:
    """
    Best-effort handling for Claude onboarding / chooser UI that can interrupt
    a normal chat flow.
    - single-choice prompts: choose the first option
    - multi-select prompts: click Skip
    """
    for selector in SEL_ASK_USER_OPTION_CANDIDATES:
        try:
            options = page.locator(selector)
            count = await options.count()
            if count == 0:
                continue
            for idx in range(count):
                option = options.nth(idx)
                try:
                    if not await option.is_visible():
                        continue
                    try:
                        await option.scroll_into_view_if_needed()
                    except Exception:
                        pass
                    option_text = ""
                    prompt_text = ""
                    all_options = []
                    try:
                        option_text = (await option.inner_text()).strip().replace("\n", " ")[:400]
                    except Exception:
                        option_text = ""
                    try:
                        listbox = option.locator("xpath=ancestor::*[@role='listbox'][1]")
                        prompt_text = (
                            (await listbox.get_attribute("aria-label"))
                            or (await listbox.inner_text())
                            or ""
                        )
                        prompt_text = prompt_text.strip().replace("\n", " ")[:600]
                        listbox_options = listbox.locator("button[role='option']")
                        option_count = await listbox_options.count()
                        for opt_idx in range(min(option_count, 8)):
                            try:
                                opt_text = (await listbox_options.nth(opt_idx).inner_text()).strip().replace("\n", " ")
                                if opt_text:
                                    all_options.append(opt_text[:300])
                            except Exception:
                                continue
                    except Exception:
                        prompt_text = ""
                    is_multi_select = False
                    for multi_sel in SEL_INTERACTIVE_CHECKBOX_CANDIDATES:
                        try:
                            if await page.locator(multi_sel).count() > 0:
                                is_multi_select = True
                                break
                        except Exception:
                            continue
                    skip_button = None
                    for skip_sel in SEL_INTERACTIVE_SKIP_CANDIDATES:
                        try:
                            candidate = page.locator(skip_sel).first
                            if await candidate.is_visible(timeout=200):
                                skip_button = candidate
                                is_multi_select = True
                                break
                        except Exception:
                            continue
                    selected_option = option_text
                    action = "select_first_option"
                    if is_multi_select and skip_button is not None:
                        action = "skip_multi_select"
                        selected_option = "Skip"
                        try:
                            await skip_button.click(timeout=1200)
                        except Exception:
                            await skip_button.click(force=True, timeout=1200)
                    else:
                        try:
                            await option.click(timeout=1200)
                        except Exception:
                            try:
                                await option.click(force=True, timeout=1200)
                            except Exception:
                                await option.focus()
                                await page.keyboard.press("Enter")
                    if on_interstitial is not None:
                        on_interstitial(
                            {
                                "event_type": "interactive_prompt",
                                "selector": selector,
                                "option_index": idx,
                                "selected_option": selected_option,
                                "interactive_action": action,
                                "is_multi_select": is_multi_select,
                                "prompt_text": prompt_text,
                                "all_options": all_options,
                                "page_url": page.url,
                                "timestamp_unix": time.time(),
                            }
                        )
                    await page.wait_for_timeout(1200)
                    return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


async def _locator_debug_snapshot(page: Page, selector: str, label: str) -> str:
    try:
        locator = page.locator(selector)
        count = await locator.count()
        visible = False
        text = ""
        if count > 0:
            try:
                visible = await locator.first.is_visible(timeout=300)
            except Exception:
                visible = False
            try:
                text = (await locator.first.inner_text()).strip().replace("\n", " ")[:160]
            except Exception:
                text = ""
        return f"{label}: selector={selector!r} count={count} visible={visible} text={text!r}"
    except Exception as exc:
        return f"{label}: selector={selector!r} ERROR={exc}"


async def _focus_chat_row_for_menu(page: Page, menu_btn) -> None:
    try:
        await menu_btn.scroll_into_view_if_needed()
    except Exception:
        pass
    try:
        box = await menu_btn.bounding_box()
        if box:
            x = max(10, box["x"] - 140)
            y = box["y"] + box["height"] / 2
            await page.mouse.click(x, y)
            await page.wait_for_timeout(120)
            return
    except Exception:
        pass


async def _hover_chat_row(page: Page, row_index: int, debug_log=None) -> bool:
    def _dbg(message: str) -> None:
        if debug_log is not None:
            debug_log(message)

    try:
        row = page.locator(SEL_CHAT_ROW).nth(row_index)
        await row.wait_for(state="visible", timeout=3000)
        try:
            label = (await row.inner_text()).strip().replace("\n", " ")[:160]
        except Exception:
            label = ""
        _dbg(f"[chat_row_hover] row_index={row_index} label={label!r}")
        await row.scroll_into_view_if_needed()
        await row.hover()
        _dbg(f"[chat_row_hover] hovered row_index={row_index}")
        await page.wait_for_timeout(180)
        return True
    except Exception as exc:
        _dbg(f"[chat_row_hover] failed for row_index={row_index}: {exc}")
        return False


async def _delete_current_chat(
    page: Page,
    retries: int = 1,
    debug_log=None,
) -> tuple[bool, str]:
    def _dbg(message: str) -> None:
        if debug_log is not None:
            debug_log(message)

    for attempt in range(1, retries + 1):
        try:
            _dbg(f"[delete_current_chat] attempt {attempt}/{retries}")
            await _hover_chat_row(page, -1, debug_log=_dbg)
            _dbg(await _locator_debug_snapshot(page, SEL_CHAT_ROW, "chat row before menu click"))
            _dbg(await _locator_debug_snapshot(page, SEL_CHAT_ROW_MENU_TRIGGER, "row-scoped chat menu trigger before click"))
            menu_btn = page.locator(SEL_CHAT_ROW_MENU_TRIGGER).last
            await menu_btn.wait_for(state="visible", timeout=5000)
            await _focus_chat_row_for_menu(page, menu_btn)
            _dbg("[delete_current_chat] clicked conversation row to surface three-dot menu")
            await menu_btn.click()
            _dbg("[delete_current_chat] clicked chat menu trigger")
            await page.wait_for_timeout(220)

            _dbg(await _locator_debug_snapshot(page, SEL_DELETE_CHAT_TRIGGER, "delete menu item after menu open"))
            delete_item = page.locator(SEL_DELETE_CHAT_TRIGGER).first
            await delete_item.wait_for(state="visible", timeout=3000)
            await delete_item.click()
            _dbg("[delete_current_chat] clicked delete menu item")
            await page.wait_for_timeout(220)

            _dbg(await _locator_debug_snapshot(page, SEL_DELETE_CONFIRM, "delete confirm button after modal open"))
            confirm_btn = page.locator(SEL_DELETE_CONFIRM).first
            await confirm_btn.wait_for(state="visible", timeout=3000)
            await confirm_btn.click()
            _dbg("[delete_current_chat] clicked delete confirm button")
            await page.wait_for_timeout(700)
            _dbg("[delete_current_chat] delete flow completed")
            return True, ""
        except Exception as exc:
            _dbg(f"[delete_current_chat] ERROR on attempt {attempt}: {exc}")
            try:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)
                _dbg("[delete_current_chat] pressed Escape after error")
            except Exception:
                pass
            if attempt >= retries:
                err = f"delete_current_chat failed after {retries} attempt(s): {exc}"
                print(f"[claude] {err}")
                return False, err
            await page.wait_for_timeout(1000)
    return False, "delete_current_chat failed without an explicit exception"


async def _ensure_sidebar_expanded(page: Page, debug_log=None) -> bool:
    """Pin the sidebar open if it's collapsed.

    On the new claude.ai layout the sidebar defaults to a narrow icon strip;
    chat rows live inside an aria-hidden / inert container until you click
    the pin-sidebar-toggle. While inert, hover/click on rows fails silently,
    which is exactly the 'removed 0 chat(s)' symptom. Returns True if the
    sidebar is open after this call (already-open or successfully toggled).
    """
    def _dbg(message: str) -> None:
        if debug_log is not None:
            debug_log(message)

    try:
        toggle = page.locator('button[data-testid="pin-sidebar-toggle"]').first
        if await toggle.count() == 0:
            _dbg("[sidebar] pin-sidebar-toggle not found; assuming already open")
            return True
        pressed = await toggle.get_attribute("aria-pressed")
        if pressed == "true":
            _dbg("[sidebar] already pinned open")
            return True
        _dbg(f"[sidebar] pinning open (was aria-pressed={pressed!r})")
        try:
            await toggle.click(timeout=1500)
        except Exception:
            await toggle.click(force=True, timeout=1500)
        await page.wait_for_timeout(500)
        # Wait for the chat-list container to drop its aria-hidden / inert state.
        for _ in range(10):
            try:
                row = page.locator(SEL_CHAT_ROW).first
                if await row.count() > 0 and await row.is_visible(timeout=300):
                    _dbg("[sidebar] chat rows visible after pin")
                    return True
            except Exception:
                pass
            await page.wait_for_timeout(300)
        _dbg("[sidebar] pinned but chat rows not yet visible — continuing anyway")
        return True
    except Exception as exc:
        _dbg(f"[sidebar] expand failed: {exc}")
        return False


async def _delete_all_chat_history(
    page: Page,
    max_deletions: int = 50,
    debug_log=None,
    on_interstitial=None,
) -> tuple[int, str]:
    def _dbg(message: str) -> None:
        if debug_log is not None:
            debug_log(message)

    deleted = 0
    last_error = ""
    await page.goto(CLAUDE_URL)
    await page.wait_for_timeout(1500)
    await _handle_claude_interstitials(page, on_interstitial=on_interstitial)
    await _ensure_sidebar_expanded(page, debug_log=_dbg)
    _dbg("[delete_all_chat_history] navigated to Claude home")

    for _ in range(max_deletions):
        try:
            rows = page.locator(SEL_CHAT_ROW)
            count = await rows.count()
            _dbg(await _locator_debug_snapshot(page, SEL_CHAT_ROW, "chat row list"))
            if count == 0:
                _dbg("[delete_all_chat_history] no chat rows found; stopping delete-all loop")
                break

            await _hover_chat_row(page, 0, debug_log=_dbg)
            _dbg(await _locator_debug_snapshot(page, SEL_CHAT_ROW_MENU_TRIGGER, "row-scoped chat menu trigger in delete-all loop"))
            target = page.locator(SEL_CHAT_ROW_MENU_TRIGGER).first
            await target.wait_for(state="visible", timeout=3000)
            await _focus_chat_row_for_menu(page, target)
            _dbg("[delete_all_chat_history] clicked first conversation row to surface three-dot menu")
            await target.click()
            _dbg("[delete_all_chat_history] clicked first chat menu trigger")
            await page.wait_for_timeout(300)

            _dbg(await _locator_debug_snapshot(page, SEL_DELETE_CHAT_TRIGGER, "delete menu item in delete-all loop"))
            delete_item = page.locator(SEL_DELETE_CHAT_TRIGGER).first
            await delete_item.wait_for(state="visible", timeout=2000)
            await delete_item.click()
            _dbg("[delete_all_chat_history] clicked delete menu item")
            await page.wait_for_timeout(300)

            _dbg(await _locator_debug_snapshot(page, SEL_DELETE_CONFIRM, "delete confirm button in delete-all loop"))
            confirm_btn = page.locator(SEL_DELETE_CONFIRM).first
            await confirm_btn.wait_for(state="visible", timeout=2000)
            await confirm_btn.click()
            _dbg("[delete_all_chat_history] clicked delete confirm button")
            await page.wait_for_timeout(1200)
            deleted += 1
            _dbg(f"[delete_all_chat_history] deleted count now {deleted}")
        except Exception as exc:
            last_error = str(exc)
            _dbg(f"[delete_all_chat_history] ERROR: {exc}")
            try:
                await page.keyboard.press("Escape")
                _dbg("[delete_all_chat_history] pressed Escape after error")
            except Exception:
                pass
            break

    return deleted, last_error


async def _check_added_memory(page: Page) -> tuple[bool, str]:
    try:
        status_badges = page.locator(SEL_ADDED_MEMORY_STATUS)
        badge_count = await status_badges.count()
        if badge_count == 0:
            return False, ""

        badge = status_badges.nth(badge_count - 1)
        badge_text = (await badge.inner_text()).strip()
        if "added memory" not in badge_text.lower():
            return False, ""

        button = page.locator(SEL_ADDED_MEMORY_BUTTON).last
        if await button.is_visible(timeout=1500):
            try:
                await button.click()
                await page.wait_for_timeout(500)
            except Exception:
                pass

        container = button.locator("xpath=ancestor::div[contains(@class,'min-w-0')][1]")
        details = ""
        try:
            details = (await container.inner_text()).strip()
        except Exception:
            pass
        if not details:
            try:
                details = (await button.locator("xpath=ancestor::div[1]").inner_text()).strip()
            except Exception:
                details = badge_text

        collapsed = re.sub(r"\s+", " ", details).strip()
        return True, collapsed[:4000]
    except Exception:
        return False, ""


async def _detect_rate_limit(page: Page) -> tuple[bool, str]:
    """Best-effort detection of Claude's rate-limit banner.

    Banner text varies — we've seen at least:
      'Usage limit reached • Resets 2:00 PM • limits shared with Claude Code'
      "You've hit your session limit • Resets at 6:00 PM"
    DOM also varies (older form has data-alert-band-wrapper, newer doesn't),
    so we try a few targeted selectors and fall back to a body-text scan.

    Returns (is_limited, snippet); snippet feeds _parse_reset_time downstream.
    """
    for selector in SEL_RATE_LIMIT_BAND_CANDIDATES:
        try:
            bands = page.locator(selector)
            count = await bands.count()
        except Exception:
            continue
        for idx in range(count):
            try:
                band_text = (await bands.nth(idx).inner_text(timeout=1500)).strip()
            except Exception:
                continue
            if not band_text:
                continue
            lowered = band_text.lower()
            if any(m in lowered for m in RATE_LIMIT_TEXT_MARKERS):
                return True, band_text[:400]

    try:
        body_text = await page.locator("body").inner_text(timeout=2000)
    except Exception:
        return False, ""
    lowered = body_text.lower()
    for marker in RATE_LIMIT_TEXT_MARKERS:
        idx = lowered.find(marker)
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(body_text), idx + 320)
            return True, body_text[start:end].strip()
    return False, ""


def _parse_reset_time(text: str, now_unix: Optional[float] = None) -> Optional[float]:
    """Parse a reset wall-clock time ("3:00 PM", "15:00") or relative duration
    ("in 2 hours", "in 45 minutes") from a rate-limit banner.

    Returns the unix timestamp at which the limit is expected to clear, or
    None if nothing parseable was found.
    """
    import datetime as _dt

    if now_unix is None:
        now_unix = time.time()

    rel_hours = re.search(r"in\s+(\d+)\s*hour", text, re.IGNORECASE)
    rel_minutes = re.search(r"(\d+)\s*minute", text, re.IGNORECASE)
    if rel_hours or (rel_minutes and re.search(r"\bin\b", text, re.IGNORECASE)):
        hours = int(rel_hours.group(1)) if rel_hours else 0
        minutes = int(rel_minutes.group(1)) if rel_minutes else 0
        if hours or minutes:
            return now_unix + hours * 3600 + minutes * 60

    abs_with_minutes = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?", text)
    abs_hour_only = re.search(r"\b(\d{1,2})\s*(AM|PM|am|pm)\b", text)

    hour: Optional[int] = None
    minute = 0
    if abs_with_minutes:
        hour = int(abs_with_minutes.group(1))
        minute = int(abs_with_minutes.group(2))
        ampm = (abs_with_minutes.group(3) or "").upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
    elif abs_hour_only:
        hour = int(abs_hour_only.group(1))
        ampm = abs_hour_only.group(2).upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0

    if hour is None:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    now_dt = _dt.datetime.fromtimestamp(now_unix)
    target = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now_dt:
        target += _dt.timedelta(days=1)
    return target.timestamp()


async def _wait_for_rate_limit_reset(
    page: Page,
    raw_banner_text: str,
    log: Optional[Callable[[str], None]] = None,
    max_wait_seconds: float = 6 * 3600,
) -> None:
    """Sleep until Claude's rate-limit banner clears.

    If a reset time can be parsed, sleep until then plus a small buffer,
    re-checking every 5 minutes in case it clears early. Otherwise fall back
    to polling every 10 minutes (with a page reload) up to max_wait_seconds.
    """
    def _say(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    started = time.time()
    deadline = started + max_wait_seconds

    reset_unix = _parse_reset_time(raw_banner_text)
    if reset_unix is not None:
        wait_secs = max(60.0, reset_unix - time.time() + 30.0)
        wait_secs = min(wait_secs, max_wait_seconds)
        eta = time.strftime("%H:%M:%S", time.localtime(time.time() + wait_secs))
        _say(
            f"[rate-limit] banner detected. Sleeping ~{wait_secs/60:.1f} min "
            f"(until ~{eta}). Banner: {raw_banner_text[:200]}"
        )
        end_wait = time.time() + wait_secs
        while time.time() < end_wait:
            chunk = min(300.0, end_wait - time.time())
            await page.wait_for_timeout(int(chunk * 1000))
            still_limited, _ = await _detect_rate_limit(page)
            if not still_limited:
                _say(f"[rate-limit] banner cleared after {(time.time()-started)/60:.1f} min — resuming")
                return
        still_limited, snippet = await _detect_rate_limit(page)
        if not still_limited:
            _say("[rate-limit] reset window passed; banner cleared — resuming")
            return
        _say(
            f"[rate-limit] reset window passed but banner still up: {snippet[:200]} "
            f"— continuing to poll every 10 min"
        )
    else:
        _say(
            f"[rate-limit] could not parse reset time from banner — polling every 10 min. "
            f"Banner: {raw_banner_text[:200]}"
        )

    while time.time() < deadline:
        await page.wait_for_timeout(int(600 * 1000))
        try:
            await page.reload()
            await _wait_for_claude_ready(page, timeout=60_000)
        except Exception:
            pass
        still_limited, _ = await _detect_rate_limit(page)
        if not still_limited:
            _say(f"[rate-limit] banner cleared after {(time.time()-started)/60:.1f} min of polling — resuming")
            return

    raise TimeoutError(
        f"[rate-limit] still present after waiting {max_wait_seconds/3600:.1f} h"
    )


async def _send_message(
    page: Page,
    text: str,
    pre_send_delay: float = 0.0,
    post_response_delay: float = 8.0,
    on_interstitial=None,
    rate_limit_log: Optional[Callable[[str], None]] = None,
) -> None:
    max_attempts = 4  # 1 happy path + up to 3 rate-limit waits
    for _ in range(max_attempts):
        is_limited, snippet = await _detect_rate_limit(page)
        if is_limited:
            await _wait_for_rate_limit_reset(page, snippet, log=rate_limit_log)
            try:
                await page.reload()
                await _wait_for_claude_ready(page, timeout=60_000)
            except Exception:
                pass
            continue

        await _handle_claude_interstitials(page, on_interstitial=on_interstitial)
        input_selector = await _wait_for_claude_ready(page, timeout=30_000)
        input_box = page.locator(input_selector).first
        await input_box.click()

        entered = False
        for sel in SEL_INPUT_CANDIDATES:
            try:
                await page.locator(sel).first.fill(text)
                entered = True
                break
            except Exception:
                pass

        if not entered:
            # Fall back to keyboard typing for contenteditable composers.
            await page.keyboard.type(text, delay=5)

        await page.wait_for_timeout(int(max(0.1, min(0.6, pre_send_delay)) * 1000))
        await _send_when_ready(page, on_interstitial=on_interstitial)

        try:
            await _wait_for_response(page, on_interstitial=on_interstitial)
        except TimeoutError:
            is_limited, snippet = await _detect_rate_limit(page)
            if is_limited:
                if rate_limit_log:
                    rate_limit_log("[rate-limit] hit during response wait — pausing and retrying")
                await _wait_for_rate_limit_reset(page, snippet, log=rate_limit_log)
                try:
                    await page.reload()
                    await _wait_for_claude_ready(page, timeout=60_000)
                except Exception:
                    pass
                continue
            raise

        is_limited, snippet = await _detect_rate_limit(page)
        if is_limited:
            if rate_limit_log:
                rate_limit_log("[rate-limit] banner appeared after send — pausing and retrying")
            await _wait_for_rate_limit_reset(page, snippet, log=rate_limit_log)
            try:
                await page.reload()
                await _wait_for_claude_ready(page, timeout=60_000)
            except Exception:
                pass
            continue

        await page.wait_for_timeout(int(post_response_delay * 1000))
        return

    raise RuntimeError(f"_send_message: rate-limit retries exhausted after {max_attempts} attempts")


async def _send_when_ready(page: Page, timeout: int = 120_000, on_interstitial=None) -> None:
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        await _handle_claude_interstitials(page, on_interstitial=on_interstitial)

        for sel in SEL_SEND_BUTTON_CANDIDATES:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=200) and await btn.is_enabled():
                    try:
                        await btn.click(timeout=1200)
                    except Exception:
                        await btn.click(force=True, timeout=1200)
                    return
            except Exception:
                pass

        stop_visible = False
        for sel in SEL_STOP_BUTTON_CANDIDATES:
            try:
                stop_btn = page.locator(sel).first
                if await stop_btn.is_visible(timeout=200):
                    stop_visible = True
                    break
            except Exception:
                pass

        if not stop_visible:
            try:
                await page.keyboard.press("Enter")
                return
            except Exception:
                pass

        await page.wait_for_timeout(600)

    await page.keyboard.press("Enter")


async def _stop_button_visible(page: Page) -> bool:
    """Returns True while Claude is still generating (Stop button is rendered).

    During Opus 4.7 extended-thinking the assistant-message bubble shows a
    short 'Thinking' summary that can stay textually stable for many seconds
    before the actual answer streams in. Watching only message-text stability
    therefore returns mid-thinking and captures the summary as the response.
    The Stop button is visible for the entire generation window (thinking +
    answer) and disappears only when the model is fully done — so we use it
    as the authoritative streaming signal.
    """
    for sel in SEL_STOP_BUTTON_CANDIDATES:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=200):
                return True
        except Exception:
            pass
    return False


async def _wait_for_response(page: Page, timeout: int = 240_000, on_interstitial=None) -> None:
    deadline = time.time() + timeout / 1000
    last_text = ""
    last_change_at = time.time()
    saw_output = False
    stop_seen = False

    while time.time() < deadline:
        await _handle_claude_interstitials(page, on_interstitial=on_interstitial)
        current_text = ""
        for sel in SEL_ASSISTANT_MESSAGE_CANDIDATES:
            try:
                msgs = page.locator(sel)
                count = await msgs.count()
                if count > 0:
                    saw_output = True
                    current_text = (await msgs.nth(count - 1).inner_text()).strip()
                    break
            except Exception:
                pass

        now = time.time()
        if current_text != last_text:
            last_text = current_text
            last_change_at = now

        stop_visible = await _stop_button_visible(page)
        if stop_visible:
            stop_seen = True
            # Reset the stability timer while Claude is still generating —
            # otherwise a long thinking-only stretch falsely looks "stable".
            last_change_at = now
            await page.wait_for_timeout(1000)
            continue

        # Stop button gone. Require it to have been seen at least once
        # (otherwise we may have polled before generation even started),
        # then a short stability window for any final text settling.
        if saw_output and stop_seen and (now - last_change_at) >= 1.5:
            return
        await page.wait_for_timeout(500)

    raise TimeoutError("Claude response timed out")


async def _get_last_response(page: Page) -> str:
    for sel in SEL_ASSISTANT_MESSAGE_CANDIDATES:
        try:
            msgs = page.locator(sel)
            count = await msgs.count()
            if count > 0:
                return (await msgs.nth(count - 1).inner_text()).strip()
        except Exception:
            pass
    return ""


async def _clear_memory(
    page: Page,
    on_interstitial=None,
    rate_limit_log: Optional[Callable[[str], None]] = None,
    max_attempts: int = 4,
) -> tuple[bool, str, str]:
    """
    Ask Claude to delete saved memory, then ask it to re-check memory status.
    Retries up to max_attempts times if confirmation does not say "ALL DELETED" —
    the model often only deletes a subset on the first call when memory is large.

    Returns:
      (memory_deleted_signal, last_deletion_reply, last_confirmation_reply)

    Important:
      This is a conversational probe, not a product-level proof. We'll still
      log it as the session's clear_memory trace because that's the behavior
      you want to evaluate and inspect.
    """
    initial_prompt = "delete everything in your saved memory"
    retry_prompt = (
        "There is still saved memory remaining. Keep calling the memory tool "
        "to delete EVERYTHING — every entry, every fact, every preference — "
        "until nothing remains. Do not stop early."
    )
    confirm_prompt = (
        "call memory tool again to check if there is still anything remained; "
        'if nothing remained, please reply "ALL DELETED"'
    )
    deleted_markers = [
        "ALL DELETED",
        "HAVE BEEN DELETED",
        "HAS BEEN DELETED",
        "NO SAVED MEMORY",
        "NO MEMORY ENTRIES",
        "NO MEMORY EDITS",
        "NOTHING REMAINED",
        "NOTHING REMAINS",
        "NO STORED MEMORY",
    ]
    failure_markers = [
        "STILL STORED",
        "REMAIN",
        "REMAINS IN MEMORY",
        "COULD NOT DELETE",
        "CANNOT DELETE",
        "UNABLE TO DELETE",
    ]

    deletion_reply = ""
    confirmation_reply = ""

    for attempt in range(1, max_attempts + 1):
        prompt = initial_prompt if attempt == 1 else retry_prompt
        await _send_message(
            page,
            prompt,
            pre_send_delay=0.4,
            post_response_delay=2.0,
            on_interstitial=on_interstitial,
            rate_limit_log=rate_limit_log,
        )
        deletion_reply = await _get_last_response(page)
        await _send_message(
            page,
            confirm_prompt,
            pre_send_delay=0.4,
            post_response_delay=2.0,
            on_interstitial=on_interstitial,
            rate_limit_log=rate_limit_log,
        )
        confirmation_reply = await _get_last_response(page)
        normalized_reply = confirmation_reply.upper()
        all_deleted = any(m in normalized_reply for m in deleted_markers) and not any(
            m in normalized_reply for m in failure_markers
        )
        if all_deleted:
            if attempt > 1 and rate_limit_log is not None:
                rate_limit_log(f"[clear_memory] confirmed ALL DELETED on attempt {attempt}/{max_attempts}")
            return True, deletion_reply, confirmation_reply
        if rate_limit_log is not None and attempt < max_attempts:
            rate_limit_log(
                f"[clear_memory] attempt {attempt}/{max_attempts} NOT CONFIRMED — "
                f"retrying. Last confirm reply: {confirmation_reply[:160]}"
            )

    return False, deletion_reply, confirmation_reply


def _log_clear_memory_result(
    log_path: Path,
    trace_jsonl_path: Path,
    *,
    sample_id: str,
    world: str,
    session_key: str,
    stage: str,
    all_deleted: bool,
    deletion_reply: str,
    confirmation_reply: str,
) -> None:
    stage_tag = f"{stage.upper()} CLEAR MEMORY"
    _append_text_line(
        log_path,
        f"{stage_tag}: "
        + ("ALL DELETED" if all_deleted else "NOT CONFIRMED")
        + (f" | reply={confirmation_reply[:160]}" if confirmation_reply else ""),
    )
    _append_jsonl(
        trace_jsonl_path,
        {
            "event_type": f"{stage}_clear_memory",
            "sample_id": sample_id,
            "world": world,
            "session_key": session_key,
            "memory_cleared_signal": all_deleted,
            "delete_prompt_reply": deletion_reply,
            "check_prompt_reply": confirmation_reply,
            "timestamp_unix": time.time(),
        },
    )


async def _run_session(
    page: Page,
    session: TestSession,
    timing: Optional[TimingProfile],
    turn_delay: float,
    history_rate: float,
    sample_id: str,
    world: str,
    conv_source: str,
    live_log_path: Optional[Path] = None,
    live_trace_jsonl_path: Optional[Path] = None,
    attempt_index: int = 1,
) -> SessionRunResult:
    results = []
    trace = []
    log_lines = []
    last_resp_len = 0
    session_memory_events = []

    def _log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)
        if live_log_path is not None:
            _append_text_line(live_log_path, message)

    def _record_trace(event: dict) -> None:
        trace.append(event)
        if live_trace_jsonl_path is not None:
            payload = dict(event)
            payload["attempt_index"] = attempt_index
            _append_jsonl(live_trace_jsonl_path, payload)

    def _record_interstitial(event: dict) -> None:
        _log(
            "    [interactive] "
            + (event.get("prompt_text", "")[:120] or "(no prompt text)")
            + " | selected="
            + (event.get("selected_option", "")[:120] or "(unknown)")
        )
        _record_trace(event)

    await _new_chat(page, on_interstitial=_record_interstitial)

    for phase in session.phases:
        for j, turn in enumerate(phase.user_turns):
            pre = timing.sample_typing_delay(len(turn)) if timing else 0.4
            post = (timing.sample_reading_delay(last_resp_len) * history_rate if timing else turn_delay)
            prefix = f"    [{phase.label}] turn {j+1}/{len(phase.user_turns)} ({len(turn)}ch pre={pre:.1f}s post={post:.1f}s)..."
            try:
                await _send_message(
                    page,
                    turn,
                    pre_send_delay=pre,
                    post_response_delay=post,
                    on_interstitial=_record_interstitial,
                    rate_limit_log=_log,
                )
                resp_text = await _get_last_response(page)
                last_resp_len = len(resp_text)
                mem_triggered, mem_content = await _check_added_memory(page)
                if mem_triggered:
                    session_memory_events.append({
                        "phase": phase.label,
                        "turn_index": j,
                        "user_turn": turn[:200],
                        "memory_content": mem_content,
                    })
                _log(prefix + (" ok [MEMORY]" if mem_triggered else " ok"))
                _record_trace({
                    "event_type": "history_turn",
                    "test_type": world,
                    "phase_label": phase.label,
                    "turn_index": j,
                    "user_input": turn,
                    "assistant_output": resp_text,
                    "pre_send_delay_sec": pre,
                    "post_response_delay_sec": post,
                    "memory_triggered": mem_triggered,
                    "memory_content": mem_content,
                })
            except Exception as e:
                _log(prefix + f" ERROR: {e}")
                _record_trace({
                    "event_type": "history_turn",
                    "test_type": world,
                    "phase_label": phase.label,
                    "turn_index": j,
                    "user_input": turn,
                    "assistant_output": "",
                    "pre_send_delay_sec": pre,
                    "post_response_delay_sec": post,
                    "memory_triggered": False,
                    "memory_content": "",
                    "error": str(e),
                })
                raise

        if not phase.mcq_items:
            continue

        _log(f"    [{phase.label}] asking {len(phase.mcq_items)} MCQs...")

        for k, item in enumerate(phase.mcq_items):
            prompt = _build_mcq_prompt(item)
            pre = timing.sample_typing_delay(len(prompt)) if timing else 0.4
            post = timing.sample_reading_delay(last_resp_len) if timing else turn_delay
            prefix = f"    MCQ {k+1}/{len(phase.mcq_items)} [{item.qa_family} {item.turn_role} {item.timestamp}] pre={pre:.1f}s..."
            rec = {
                "sample_id": sample_id,
                "world": world,
                "session_key": session.session_key,
                "phase_label": phase.label,
                "conv_source": conv_source,
                "qa_family": item.qa_family,
                "timestamp": item.timestamp,
                "turn_role": item.turn_role,
                "identifier_label": item.identifier_label,
                "sensitive_key": item.sensitive_key,
                "sensitive_value": item.sensitive_value,
                "question": item.question,
                "choices": item.choices,
                "choice_to_answer_type": item.choice_to_answer_type,
                "correct_choice": item.correct_choice,
                "num_history_turns": sum(len(p.user_turns) for p in session.phases),
                "session_memory_events": list(session_memory_events),
                "model_response": "",
                "predicted_choice": "",
                "predicted_answer_type": "",
                "error": None,
            }
            try:
                await _send_message(
                    page,
                    prompt,
                    pre_send_delay=pre,
                    post_response_delay=post,
                    on_interstitial=_record_interstitial,
                    rate_limit_log=_log,
                )
                response_text = await _get_last_response(page)
                last_resp_len = len(response_text)
                predicted = _extract_choice(response_text, item.choice_order)
                predicted_type = item.choice_to_answer_type.get(predicted, "")
                correct_type = item.choice_to_answer_type.get(item.correct_choice, "")
                rec["model_response"] = response_text
                rec["predicted_choice"] = predicted
                rec["predicted_answer_type"] = predicted_type
                mem_triggered, mem_content = await _check_added_memory(page)
                if mem_triggered:
                    session_memory_events.append({
                        "phase": phase.label,
                        "turn_index": f"mcq_{k}",
                        "user_turn": prompt[:200],
                        "memory_content": mem_content,
                    })
                    rec["session_memory_events"] = list(session_memory_events)
                _log(
                    prefix + " " +
                    f"{'✓' if predicted == item.correct_choice else '✗'} pred={predicted} "
                    f"[pred_type={predicted_type}] correct={item.correct_choice} "
                    f"[correct_type={correct_type}]" +
                    (" [MEMORY]" if mem_triggered else "")
                )
                _record_trace({
                    "event_type": "mcq",
                    "test_type": world,
                    "phase_label": phase.label,
                    "mcq_index": k,
                    "qa_family": item.qa_family,
                    "turn_role": item.turn_role,
                    "timestamp": item.timestamp,
                    "question": item.question,
                    "user_input": prompt,
                    "assistant_output": response_text,
                    "predicted_choice": predicted,
                    "predicted_answer_type": predicted_type,
                    "correct_choice": item.correct_choice,
                    "correct_answer_type": correct_type,
                    "pre_send_delay_sec": pre,
                    "post_response_delay_sec": post,
                    "memory_triggered": mem_triggered,
                    "memory_content": mem_content,
                })
            except Exception as e:
                rec["error"] = str(e)
                _log(prefix + f" ERROR: {e}")
                _record_trace({
                    "event_type": "mcq",
                    "test_type": world,
                    "phase_label": phase.label,
                    "mcq_index": k,
                    "qa_family": item.qa_family,
                    "turn_role": item.turn_role,
                    "timestamp": item.timestamp,
                    "question": item.question,
                    "user_input": prompt,
                    "assistant_output": "",
                    "predicted_choice": "",
                    "predicted_answer_type": "",
                    "correct_choice": item.correct_choice,
                    "correct_answer_type": item.choice_to_answer_type.get(item.correct_choice, ""),
                    "pre_send_delay_sec": pre,
                    "post_response_delay_sec": post,
                    "memory_triggered": False,
                    "memory_content": "",
                    "error": str(e),
                })
            results.append(rec)

    return SessionRunResult(records=results, trace=trace, log_lines=log_lines)


async def login_only(args: argparse.Namespace) -> None:
    print(f"Login-only mode. Claude session will be saved to: {args.session_dir}")
    async with async_playwright() as pw:
        context: BrowserContext = await pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.session_dir).resolve()),
            headless=False,
            channel="chrome",
            args=[
                "--window-size=1280,900",
                "--disable-features=DnsOverHttps,EncryptedClientHello",
            ],
            ignore_default_args=["--enable-automation", "--no-sandbox"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await context.add_init_script(
            "try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}"
        )
        try:
            await page.goto(CLAUDE_URL)
        except Exception as exc:
            print(f"WARNING: initial goto failed: {exc}")
            print("The browser will stay open — try navigating to https://claude.ai/ manually.")
        print("Complete Claude login in the browser window.")
        print("Press Enter here when you are fully logged in and see the main chat UI...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "")
        try:
            sel = await _wait_for_claude_ready(page, timeout=15_000)
            print(f"Claude ready. Detected input selector: {sel}")
        except Exception as exc:
            print(f"WARNING: Claude input was not confirmed after Enter: {exc}")
        print("Session saved. You can now run the main evaluation.")
        await context.close()


async def _wait_for_user_after_ready(label: str) -> None:
    loop = asyncio.get_event_loop()
    print(label)
    await loop.run_in_executor(None, input, "")


async def _prompt_manual_delete(page: Page, label: str) -> None:
    loop = asyncio.get_event_loop()
    print(f"{label}: manually delete the current Claude chat in the browser, then press Enter to continue...")
    await loop.run_in_executor(None, input, "")
    try:
        await page.goto(CLAUDE_URL)
        await _wait_for_claude_ready(page, timeout=60_000)
        await page.wait_for_timeout(1500)
    except Exception:
        print("WARNING: Claude input did not reappear after manual delete. Check the browser state.")


async def evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    run_debug_dir = output_root.parent / "claude_web_results" / "_run_debug"
    run_debug_dir.mkdir(parents=True, exist_ok=True)

    all_samples = _discover_samples(args.data_dir, args.topic)
    if args.sample_id_filter:
        all_samples = [s for s in all_samples if s.startswith(args.sample_id_filter)]
    if args.limit:
        all_samples = all_samples[: args.limit]

    print(f"Topic={args.topic}  World={args.world}  Personas={len(all_samples)}")

    timing = None
    if args.timing_profile and Path(args.timing_profile).exists():
        timing = TimingProfile.load(args.timing_profile)
        print(f"Using human timing profile ({timing.n_interactions} interactions)")
    elif args.timing_profile:
        print("WARNING: timing profile not found — using fixed delays")

    persona_output_paths = []

    async with async_playwright() as pw:
        context: BrowserContext = await pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.session_dir).resolve()),
            headless=args.headless,
            channel="chrome",
            args=[
                "--window-size=1280,900",
                "--disable-features=DnsOverHttps,EncryptedClientHello",
            ],
            ignore_default_args=["--enable-automation", "--no-sandbox", "--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await context.add_init_script(
            "try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}"
        )

        try:
            await page.goto(CLAUDE_URL)
        except Exception as exc:
            print(f"WARNING: initial goto failed: {exc}")
            print("The browser will stay open — try navigating to https://claude.ai/ manually.")
        print("If you see a login page, complete it now (up to 5 minutes).")
        try:
            selector = await _wait_for_claude_ready(page, timeout=300_000)
            print(f"Ready. Detected input selector: {selector}")
            if args.wait_for_ready_enter:
                await _wait_for_user_after_ready(
                    "Claude is ready. Press Enter here when you want the evaluation to begin..."
                )
        except Exception as exc:
            print(f"WARNING: Claude did not load — check the browser window. {exc}")

        try:
            for i, sample_id in enumerate(all_samples):
                print(f"\n[{i+1}/{len(all_samples)}] {sample_id}")
                try:
                    conv, conv_source = _load_conv(sample_id, args.data_dir, args.world)
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                    continue

                mcq_items = _load_mcq_items(args.data_dir, args.topic, sample_id)
                if not mcq_items:
                    print("  SKIP: no MCQ items found")
                    continue

                sessions = _plan_sessions(sample_id, conv, args.world, mcq_items)
                persona_output_path = _claude_persona_output_path(output_root, args.topic, args.world, sample_id)
                persona_output_path.parent.mkdir(parents=True, exist_ok=True)
                persona_output_paths.append(persona_output_path)
                print(f"  {len(sessions)} session(s) planned: {[s.session_key for s in sessions]}")

                for sess in sessions:
                    artifacts = _claude_session_artifact_paths(output_root, args.topic, args.world, sample_id, sess.session_key)
                    if not args.overwrite and _session_is_completed(artifacts["result"]):
                        print(f"  [{sess.session_key}] already done, skipping")
                        continue

                    phase_summary = " -> ".join(
                        f"{p.label}({len(p.user_turns)}turns" + (f"+{len(p.mcq_items)}MCQs" if p.mcq_items else "") + ")"
                        for p in sess.phases
                    )
                    print(f"  [{sess.session_key}] {phase_summary}")
                    artifacts["dir"].mkdir(parents=True, exist_ok=True)
                    artifacts["debug_dir"].mkdir(parents=True, exist_ok=True)
                    artifacts["log"].write_text("", encoding="utf-8")
                    artifacts["trace_jsonl"].write_text("", encoding="utf-8")

                    def _record_pre_session_interstitial(event: dict) -> None:
                        _append_text_line(
                            artifacts["log"],
                            "PRE-SESSION INTERACTIVE: "
                            + (event.get("prompt_text", "")[:120] or "(no prompt text)")
                            + " | selected="
                            + (event.get("selected_option", "")[:120] or "(unknown)"),
                        )
                        payload = dict(event)
                        payload.update({
                            "sample_id": sample_id,
                            "world": args.world,
                            "session_key": sess.session_key,
                            "phase_label": "pre_session_cleanup",
                        })
                        _append_jsonl(artifacts["trace_jsonl"], payload)

                    def _pre_session_rate_limit_log(message: str) -> None:
                        print(message, flush=True)
                        _append_text_line(artifacts["log"], message)

                    print("  Pre-session cleanup: clearing Claude memory...", end=" ", flush=True)
                    pre_all_deleted = False
                    pre_deletion_reply = ""
                    pre_confirmation_reply = ""
                    try:
                        pre_all_deleted, pre_deletion_reply, pre_confirmation_reply = await _clear_memory(
                            page,
                            on_interstitial=_record_pre_session_interstitial,
                            rate_limit_log=_pre_session_rate_limit_log,
                        )
                        print("ALL DELETED" if pre_all_deleted else "NOT CONFIRMED", flush=True)
                    except Exception as exc:
                        pre_confirmation_reply = f"pre_session_clear_memory_error: {exc}"
                        print("ERROR", flush=True)
                    _log_clear_memory_result(
                        artifacts["log"],
                        artifacts["trace_jsonl"],
                        sample_id=sample_id,
                        world=args.world,
                        session_key=sess.session_key,
                        stage="pre_session",
                        all_deleted=pre_all_deleted,
                        deletion_reply=pre_deletion_reply,
                        confirmation_reply=pre_confirmation_reply,
                    )
                    if pre_confirmation_reply.startswith("pre_session_clear_memory_error:"):
                        debug_paths = await _capture_debug_artifacts(
                            page,
                            artifacts["debug_dir"],
                            0,
                            "pre_session_clear_memory_error",
                        )
                        _append_text_line(
                            artifacts["log"],
                            f"PRE-SESSION CLEAR MEMORY DEBUG: {json.dumps(debug_paths, ensure_ascii=False)}",
                        )
                        _append_jsonl(
                            artifacts["trace_jsonl"],
                            {
                                "event_type": "pre_session_clear_memory_debug",
                                "sample_id": sample_id,
                                "world": args.world,
                                "session_key": sess.session_key,
                                "debug_paths": debug_paths,
                                "timestamp_unix": time.time(),
                            },
                        )

                    print("  Pre-session cleanup: deleting all Claude chat history...", end=" ", flush=True)
                    deleted_count, delete_all_error = await _delete_all_chat_history(
                        page,
                        debug_log=lambda message: _append_text_line(artifacts["log"], message),
                        on_interstitial=_record_pre_session_interstitial,
                    )
                    print(f"removed {deleted_count} chat(s)", flush=True)
                    _append_text_line(
                        artifacts["log"],
                        f"PRE-SESSION DELETE ALL: removed {deleted_count} chat(s)"
                        + (f" | last_error={delete_all_error}" if delete_all_error else ""),
                    )
                    _append_jsonl(
                        artifacts["trace_jsonl"],
                        {
                            "event_type": "pre_session_delete_all",
                            "sample_id": sample_id,
                            "world": args.world,
                            "session_key": sess.session_key,
                            "deleted_count": deleted_count,
                            "last_error": delete_all_error,
                            "timestamp_unix": time.time(),
                        },
                    )
                    if delete_all_error:
                        debug_paths = await _capture_debug_artifacts(
                            page,
                            artifacts["debug_dir"],
                            0,
                            "pre_session_delete_all_error",
                        )
                        _append_text_line(
                            artifacts["log"],
                            f"PRE-SESSION DELETE ALL DEBUG: {json.dumps(debug_paths, ensure_ascii=False)}",
                        )
                        _append_jsonl(
                            artifacts["trace_jsonl"],
                            {
                                "event_type": "pre_session_delete_all_debug",
                                "sample_id": sample_id,
                                "world": args.world,
                                "session_key": sess.session_key,
                                "debug_paths": debug_paths,
                                "timestamp_unix": time.time(),
                            },
                        )

                    max_attempts = max(1, args.session_retries + 1)
                    session_result = None
                    session_error = ""
                    session_status = "error"

                    for attempt_idx in range(1, max_attempts + 1):
                        _append_text_line(artifacts["log"], f"=== Attempt {attempt_idx}/{max_attempts} for session {sess.session_key} ===")
                        try:
                            session_result = await _run_session(
                                page,
                                sess,
                                timing,
                                args.turn_delay,
                                args.history_rate,
                                sample_id,
                                args.world,
                                conv_source,
                                live_log_path=artifacts["log"],
                                live_trace_jsonl_path=artifacts["trace_jsonl"],
                                attempt_index=attempt_idx,
                            )
                            session_status = "completed"
                            session_error = ""
                            break
                        except Exception as e:
                            session_error = str(e)
                            print(f"  SESSION ERROR (attempt {attempt_idx}/{max_attempts}): {e}", flush=True)
                            await _capture_debug_artifacts(page, artifacts["debug_dir"], attempt_idx, "session_error")
                            try:
                                await page.goto(CLAUDE_URL)
                                await page.wait_for_timeout(3000)
                            except Exception:
                                pass
                            if attempt_idx >= max_attempts:
                                session_result = SessionRunResult(
                                    records=[{
                                        "sample_id": sample_id,
                                        "world": args.world,
                                        "session_key": sess.session_key,
                                        "error": session_error,
                                    }],
                                    trace=[],
                                    log_lines=[f"SESSION ERROR after {max_attempts} attempt(s): {session_error}"],
                                )
                                break

                    _write_claude_session_artifacts(
                        output_root,
                        args.topic,
                        args.world,
                        sample_id,
                        sess,
                        conv_source,
                        session_result,
                        status=session_status,
                        error=session_error,
                    )
                    with persona_output_path.open("a", encoding="utf-8") as out_f:
                        for rec in session_result.records:
                            out_f.write(f"{__import__('json').dumps(rec, ensure_ascii=False)}\n")

                    delay = timing.sample_reading_delay(200) if timing else args.sample_delay
                    await page.wait_for_timeout(int(delay * 1000))
        finally:
            await context.close()

    print(f"\nDone. Persona-level results saved under {_claude_results_root(output_root)}")
    seen = set()
    for path in persona_output_paths:
        if path in seen:
            continue
        seen.add(path)
        print(f"  {path}")
        _print_summary(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Claude web on MemoryCtrl recall MCQs via Playwright")
    parser.add_argument("--topic", default="travelPlanning")
    parser.add_argument("--world", default="baseline", choices=["baseline", "forget", "no_store"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default="results/claude_web_baseline.jsonl")
    parser.add_argument("--timing_profile", default=DEFAULT_TIMING_PROFILE)
    parser.add_argument("--turn_delay", type=float, default=8.0)
    parser.add_argument("--history_rate", type=float, default=0.2)
    parser.add_argument("--sample_delay", type=float, default=5.0)
    parser.add_argument("--session_dir", default="./claude_session")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--login", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_id_filter", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--session_retries", type=int, default=1)
    parser.add_argument("--wait_for_ready_enter", action="store_true")
    args = parser.parse_args()
    if args.login:
        asyncio.run(login_only(args))
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
