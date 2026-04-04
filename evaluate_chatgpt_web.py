#!/usr/bin/env python3
"""
evaluate_chatgpt_web.py

Automates ChatGPT web UI (chatgpt.com) to evaluate PersonaMem recall MCQs.

Setup:
  pip install playwright
  playwright install chromium

First run opens the browser for manual login; the session is saved to
./chatgpt_session/ so subsequent runs skip login.

==================================================================
NOTE: Within-conversation testing only (current scope)
==================================================================
This script tests the WITHIN-CONVERSATION case: the entire persona
conversation history is replayed inside a single ChatGPT conversation,
and MCQs are asked in the same conversation.

TODO (cross-session testing):
  A future variant should test the CROSS-SESSION case, where each stage
  of the conversation is sent in a separate ChatGPT session (relying on
  ChatGPT's persistent Memory feature to carry information across sessions).
  This requires:
    - NOT clearing persistent memory between stages
    - Clearing memory only between personas
    - A different session plan that maps stages to separate browser sessions

==================================================================
Session design per world (within-conversation)
==================================================================

BASELINE — 1 session per persona
  Feed:  all user turns (up to Late Stage)
  Ask:   all whole_recall + slot_recall MCQs, shuffled

NO_STORE — 3 sessions per persona (clear memory + delete chat before each)
  Session 1: feed Initial+Early           → ask ALL MCQs (shuffled)
  Session 2: feed Initial+Early+Intermed. → ask ALL MCQs (shuffled)
  Session 3: feed Initial+Early+Intermed.+Late → ask ALL MCQs (shuffled)
  Memory is cleared and chat is deleted between sessions via the settings UI.

FORGET — 1 interleaved session per persona
  The 3 key turns each have a forget instruction in a different stage.
  A single ChatGPT conversation is used; history and MCQs are interleaved:

    Send all Initial Stage user turns
    Send all Early Stage user turns        ← contains forget(key1)
    Ask: key1 whole+slot MCQs + probe1 (shuffled)
    Send all Intermediate Stage user turns ← contains forget(key2)
    Ask: key2 whole+slot MCQs + probe2 (shuffled)
    Send all Late Stage user turns         ← contains forget(key3)
    Ask: key3 whole+slot MCQs + probe3 (shuffled)

  Probes are assigned one per stage in chronological order.

MEMORY CLEARING
  After every test session (baseline, each no_store session, forget),
  the script: (1) clears ChatGPT persistent memory via settings UI,
  then (2) deletes the current chat conversation.
  If either UI step fails, a warning is printed — handle manually.

==================================================================
File paths (relative to --data_dir, default: data)
==================================================================
  baseline → data/baseline/<topic>/conversation_<sample_id>.json
  no_store → data/test/<topic>/no_store/transformed_histories/
               conversation_<sample_id>.no_store.transformed_history.json
  forget   → data/test/<topic>/forget/transformed_histories/
               conversation_<sample_id>.forget.transformed_history.json
  MCQs     → data/test/<topic>/whole_recall/whole_recall_qa_<sample_id>.json
             data/test/<topic>/slot_recall/slot_recall_qa_<sample_id>.json

==================================================================
Usage
==================================================================
  python evaluate_chatgpt_web.py --topic travelPlanning --world baseline \\
    --output results/chatgpt_web_baseline.jsonl

  python evaluate_chatgpt_web.py --topic travelPlanning --world no_store \\
    --output results/chatgpt_web_no_store.jsonl

  python evaluate_chatgpt_web.py --topic travelPlanning --world forget \\
    --output results/chatgpt_web_forget.jsonl

  # single persona test run
  python evaluate_chatgpt_web.py --topic travelPlanning --world baseline \\
    --sample_id_filter travelPlanning_persona0 \\
    --output results/chatgpt_web_test.jsonl

Outputs:
  `--output` is treated as a root hint, not as one big aggregate file.
  Actual outputs are written per persona under:
  `<output_parent>/chatgpt_web_results/<topic>/<sample_id>/test_type_<world>/`
  including:
    - `results.jsonl`          : MCQ-level records for that persona+test_type
  Each completed test session also writes its own directory under:
  `<output_parent>/chatgpt_web_results/<topic>/<sample_id>/test_type_<world>/session_<session_key>/`
  containing:
    - `session_result.json`  : session-level summary plus MCQ records
    - `session_trace.json`   : round-by-round user input and ChatGPT output
    - `session_log.txt`      : the same per-session log text printed to terminal

  Resume behavior uses `session_result.json` as the source of truth: if that
  file already exists for a given `(topic, world, sample_id, session_key)`,
  the session is skipped.

Manual cleanup:
  Pass `--manual_cleanup` to pause before pre-run cleanup and before each
  session cleanup. In that mode, you manually delete memory and delete the
  current chat in the browser, then return to the main chat page and press
  Enter to continue.
"""

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from patchright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PWTimeout


# ---------------------------------------------------------------------------
# Human timing profile
# ---------------------------------------------------------------------------

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
        min_cps = max(1.0, self.typing_chars_per_sec_mean * 0.3)
        cps = max(min_cps, random.gauss(self.typing_chars_per_sec_mean,
                                        self.typing_chars_per_sec_std))
        pause = max(0.2, random.gauss(self.pre_send_pause_mean, self.pre_send_pause_std))
        return min(30.0, msg_len / cps + pause)

    def sample_reading_delay(self, response_len: int) -> float:
        rps = max(5.0, random.gauss(self.reading_chars_per_sec_mean,
                                     self.reading_chars_per_sec_std))
        thinking = max(0.5, random.gauss(self.post_reading_pause_mean,
                                          self.post_reading_pause_std))
        return max(self.min_turn_delay, min(self.max_turn_delay,
                                             response_len / rps + thinking))


# ---------------------------------------------------------------------------
# ChatGPT selectors — update here if the UI changes
# ---------------------------------------------------------------------------
CHATGPT_URL   = "https://chatgpt.com/"
SEL_INPUT     = "#prompt-textarea"
SEL_SEND_BTN  = 'button[data-testid="send-button"]'
SEL_STOP_BTN  = 'button[aria-label="Stop streaming"]'
SEL_NEW_CHAT  = 'a[data-testid="new-conversation-button"], nav a[href="/"]'
SEL_ASST_MSG  = '[data-message-author-role="assistant"]'
SEL_LOGGED_IN = 'nav, [data-testid="profile-button"], #sidebar-desktop-content'

# Settings UI selectors (verified from DevTools)
SEL_PROFILE_BTN       = '[data-testid="accounts-profile-button"]'
SEL_SETTINGS_LNK      = '[data-testid="settings-menu-item"]'
SEL_PERSONAL_TAB      = 'button:has-text("Personalization"), [role="tab"]:has-text("Personalization"), div[role="menuitem"]:has-text("Personalization")'
SEL_MEMORY_MANAGE_BTN = 'button[aria-label="Manage memories"]'
# ⋯ button in Saved memories dialog: aria-label="More options"
SEL_MEMORY_KEBAB      = 'button[aria-label="More options"]'
SEL_MEMORY_MENU       = '[role="menu"][data-state="open"], [role="menu"][data-state="open"] *'
# "Delete all memories" menuitem (red text)
SEL_MEMORY_DELETE_ALL = 'div[role="menuitem"]:has-text("Delete all memories"), [role="menuitem"]:has-text("Delete all memories")'
# Confirmation buttons (specific per dialog)
SEL_CONFIRM_DELETE_CHATS   = '[data-testid="confirm-delete-all-chats-button"]'
SEL_CONFIRM_DELETE_MEMORIES = 'button.btn-danger:has-text("Delete all"), button:has-text("Delete all")'
# Data controls tab and delete-all button
SEL_DATA_CONTROLS_TAB = '[data-testid="data-controls-tab"], button:has-text("Data controls"), [role="tab"]:has-text("Data controls")'
SEL_DELETE_ALL_CHATS  = 'button[aria-label*="Delete all chats"]'

# ---------------------------------------------------------------------------
# Conversation stages
# ---------------------------------------------------------------------------
PERIODS = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]
PERIOD_SHORT = {
    "Conversation Initial Stage":      "initial",
    "Conversation Early Stage":        "early",
    "Conversation Intermediate Stage": "intermediate",
    "Conversation Late Stage":         "late",
}

# ---------------------------------------------------------------------------
# MCQ data structures
# ---------------------------------------------------------------------------

@dataclass
class McqItem:
    sample_id: str
    qa_family: str         # "whole_recall" | "slot_recall"
    timestamp: str
    turn_role: str         # "key" | "probe"
    identifier_label: str
    user_turn: str
    sensitive_key: str     # slot_recall only
    sensitive_value: str   # slot_recall only
    question: str
    choices: dict
    choice_order: list
    choice_to_answer_type: dict
    correct_choice: str


def _load_mcq_items(data_dir: str, topic: str, sample_id: str) -> list[McqItem]:
    items: list[McqItem] = []
    root = Path(data_dir) / "test" / topic

    wr_path = root / "whole_recall" / f"whole_recall_qa_{sample_id}.json"
    if wr_path.exists():
        for item in json.loads(wr_path.read_text(encoding="utf-8")).get("items", []):
            r = item["rendered"]
            items.append(McqItem(
                sample_id=sample_id, qa_family="whole_recall",
                timestamp=item["timestamp"], turn_role=item.get("turn_role", ""),
                identifier_label=item.get("identifier_label", ""),
                user_turn=item.get("user_turn", ""),
                sensitive_key="", sensitive_value="",
                question=r["question"], choices=r["choices"],
                choice_order=r["choice_order"],
                choice_to_answer_type=r["choice_to_answer_type"],
                correct_choice=r["remember_correct_choice"],
            ))

    sr_path = root / "slot_recall" / f"slot_recall_qa_{sample_id}.json"
    if sr_path.exists():
        for item in json.loads(sr_path.read_text(encoding="utf-8")).get("items", []):
            for slot in item["rendered"]["items"]:
                items.append(McqItem(
                    sample_id=sample_id, qa_family="slot_recall",
                    timestamp=item["timestamp"], turn_role=item.get("turn_role", ""),
                    identifier_label=item.get("identifier_label", ""),
                    user_turn=item.get("user_turn", ""),
                    sensitive_key=slot["sensitive_key"],
                    sensitive_value=slot["sensitive_value"],
                    question=slot["question"], choices=slot["choices"],
                    choice_order=slot["choice_order"],
                    choice_to_answer_type=slot["choice_to_answer_type"],
                    correct_choice=slot["remember_correct_choice"],
                ))

    return items


def _discover_samples(data_dir: str, topic: str) -> list[str]:
    wr_dir = Path(data_dir) / "test" / topic / "whole_recall"
    prefix = f"whole_recall_qa_{topic}_"
    return sorted(
        f"{topic}_{p.stem[len(prefix):]}" for p in wr_dir.glob(f"{prefix}*.json")
    )


# ---------------------------------------------------------------------------
# Conversation loading helpers
# ---------------------------------------------------------------------------

def _load_conv(sample_id: str, data_dir: str, world: str) -> tuple[dict, str]:
    """Load and return (conv_dict, source_path_str)."""
    topic = sample_id.split("_")[0]
    root = Path(data_dir)
    if world == "baseline":
        path = root / "baseline" / topic / f"conversation_{sample_id}.json"
    else:
        path = (
            root / "test" / topic / world / "transformed_histories"
            / f"conversation_{sample_id}.{world}.transformed_history.json"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Conversation not found for {sample_id} (world={world}):\n  {path}"
        )
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _stage_user_turns(conv: dict, stage: str) -> list[str]:
    """User turns for a single stage (no cumulation)."""
    return [
        line[5:].strip()
        for line in conv.get(stage, [])
        if isinstance(line, str) and line.startswith("User:")
    ]


def _cumulative_user_turns(conv: dict, up_to_period: str) -> list[str]:
    """User turns from Initial Stage up to and including up_to_period."""
    end_idx = PERIODS.index(up_to_period)
    turns = []
    for period in PERIODS[: end_idx + 1]:
        turns.extend(_stage_user_turns(conv, period))
    return turns


# ---------------------------------------------------------------------------
# Forget stage detection
# ---------------------------------------------------------------------------

def _detect_forget_stage_per_key(
    conv: dict, mcq_items: list[McqItem]
) -> dict[str, str]:
    """
    Returns {key_timestamp: forget_stage} by scanning each stage
    for "Please forget/clear/remove" instruction lines, then assigning
    stages to keys in chronological key order.
    """
    forget_kw = ("please forget", "please clear", "please remove")
    forget_stages: list[str] = []
    for stage in PERIODS:
        for line in conv.get(stage, []):
            if (isinstance(line, str) and line.startswith("User:") and
                    any(kw in line.lower() for kw in forget_kw)):
                forget_stages.append(stage)
                break  # one forget instruction per stage is enough

    key_timestamps = sorted(
        {item.timestamp for item in mcq_items if item.turn_role == "key"}
    )
    return {ts: stage for ts, stage in zip(key_timestamps, forget_stages)}


# ---------------------------------------------------------------------------
# Session plan data structures
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """A batch of user turns to feed, optionally followed by MCQ questions."""
    label: str                          # human-readable identifier
    user_turns: list[str]               # turns to send in this phase
    mcq_items: list[McqItem] = field(default_factory=list)  # questions after turns


@dataclass
class TestSession:
    """One ChatGPT conversation = one test session."""
    session_key: str          # unique id used for resume tracking
    phases: list[Phase]       # ordered list of (feed → ask) phases


@dataclass
class SessionRunResult:
    records: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    log_lines: list[str]


# ---------------------------------------------------------------------------
# Session planning
# ---------------------------------------------------------------------------

def _plan_sessions(
    sample_id: str,
    conv: dict,
    world: str,
    mcq_items: list[McqItem],
) -> list[TestSession]:

    all_mcq = list(mcq_items)

    # ---- BASELINE: one session, full history, all MCQs ----
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

    # ---- NO_STORE: 3 separate sessions ----
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

    # ---- FORGET: 1 interleaved session ----
    if world == "forget":
        forget_stage_map = _detect_forget_stage_per_key(conv, mcq_items)
        if not forget_stage_map:
            print(f"  WARNING: no forget stages detected — skipping forget session")
            return []

        # Group MCQ items by timestamp
        by_ts: dict[str, list[McqItem]] = {}
        for item in mcq_items:
            by_ts.setdefault(item.timestamp, []).append(item)

        # Collect probe turn groups (all MCQs for each probe timestamp)
        probe_ts_ordered = sorted(
            {item.timestamp for item in mcq_items if item.turn_role == "probe"}
        )
        probe_groups = [
            [i for i in by_ts.get(ts, []) if i.turn_role == "probe"]
            for ts in probe_ts_ordered
        ]

        # Build phases: Initial Stage has no MCQs; each subsequent stage has
        # the key MCQs + one assigned probe group.
        phases: list[Phase] = [
            Phase(
                label="initial",
                user_turns=_stage_user_turns(conv, "Conversation Initial Stage"),
                mcq_items=[],   # no questions yet
            )
        ]

        key_ts_ordered = sorted(forget_stage_map.keys())
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


# ---------------------------------------------------------------------------
# MCQ prompt
# ---------------------------------------------------------------------------

def _build_mcq_prompt(item: McqItem) -> str:
    options = "\n".join(
        f"({lbl.lower()}) {item.choices[lbl]}" for lbl in item.choice_order
    )
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


# ---------------------------------------------------------------------------
# Browser / ChatGPT helpers
# ---------------------------------------------------------------------------


async def _new_chat(page: Page) -> None:
    try:
        btn = page.locator(SEL_NEW_CHAT).first
        if await btn.is_visible(timeout=3000):
            await btn.click()
            await page.wait_for_timeout(1500)
            return
    except Exception:
        pass
    await page.goto(CHATGPT_URL)
    await page.wait_for_timeout(2000)


async def _open_settings_dialog(page: Page) -> bool:
    """
    Navigate to chatgpt.com, find the profile button, open the Settings dialog.
    Returns True when the Settings dialog is open, False on failure.
    Does NOT use JS-click-last-button fallback (would click Upgrade instead).
    """
    # Only navigate if we're not already on chatgpt.com with the input ready
    already_ready = False
    try:
        already_ready = await page.locator(SEL_INPUT).is_visible(timeout=1000)
    except Exception:
        pass
    if not already_ready:
        await page.goto(CHATGPT_URL)
        print("Waiting for ChatGPT to be ready (complete any login/verification if needed)...")
        try:
            await page.wait_for_selector(SEL_INPUT, timeout=300_000)
            print("ChatGPT ready.")
        except PWTimeout:
            print("WARNING: ChatGPT did not load in time — settings step will be skipped.")
            return False
    await page.wait_for_timeout(1500)

    # Scroll sidebar to bottom so profile button is visible
    await page.evaluate("""
        () => {
            const nav = document.querySelector('nav') ||
                        document.querySelector('#sidebar-desktop-content');
            if (nav) nav.scrollTop = nav.scrollHeight;
        }
    """)
    await page.wait_for_timeout(500)

    # Debug: print all nav buttons to help diagnose selector issues
    nav_btns = await page.evaluate("""
        () => [...document.querySelectorAll('nav button')].map(b =>
            JSON.stringify({text: b.innerText.trim().slice(0,40),
                            testid: b.dataset.testid || '',
                            aria: b.getAttribute('aria-label') || ''}))
    """)
    print(f"[debug] nav buttons: {nav_btns}")

    # Try known selectors for the profile/account button
    profile_clicked = False
    for sel in [
        '[data-testid="accounts-profile-button"]',
        '[data-testid="profile-button"]',
        'button[aria-label*="open profile menu" i]',
        'button[aria-label*="profile" i]',
    ]:
        try:
            el = page.locator(sel).last
            if await el.is_visible(timeout=2000):
                await el.click()
                profile_clicked = True
                print(f"[debug] clicked profile via: {sel}")
                break
        except Exception:
            pass

    if not profile_clicked:
        print("[debug] profile button not found by any selector")
        return False

    await page.wait_for_timeout(700)

    # Click "Settings" in the profile menu
    try:
        s = page.locator(SEL_SETTINGS_LNK).first
        if not await s.is_visible(timeout=2000):
            return False
        await s.click()
        await page.wait_for_timeout(1000)
        return True
    except Exception:
        return False


async def _delete_current_chat(page: Page, retries: int = 3) -> bool:
    """
    Settings → Data controls → Delete all chats.
    If there are no chats (button absent/disabled), returns True (nothing to do).
    """
    for attempt in range(1, retries + 1):
        try:
            if not await _open_settings_dialog(page):
                raise RuntimeError("could not open Settings dialog")

            dc = page.locator(SEL_DATA_CONTROLS_TAB).first
            if not await dc.is_visible(timeout=2000):
                raise RuntimeError("Data controls tab not found")
            await dc.click()
            await page.wait_for_timeout(700)

            del_btn = page.locator(SEL_DELETE_ALL_CHATS).first
            if not await del_btn.is_visible(timeout=2000):
                # No chats to delete — that's fine
                await page.keyboard.press("Escape")
                return True
            await del_btn.click()
            await page.wait_for_timeout(600)

            try:
                confirm = page.locator(SEL_CONFIRM_DELETE_CHATS).first
                if await confirm.is_visible(timeout=2000):
                    await confirm.click()
                    await page.wait_for_timeout(600)
            except Exception:
                pass

            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
            return True

        except Exception as e:
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass
            if attempt < retries:
                print(f"[delete chat failed (attempt {attempt}/{retries}): {e}] retrying in 10s...")
                await asyncio.sleep(10)
            else:
                print(f"[delete chat failed after {retries} attempts: {e}]")
    return False


async def _open_settings_personalization(page: Page) -> bool:
    """
    Open ChatGPT Settings → Personalization tab.
    Returns True when the Personalization tab is active.
    """
    if not await _open_settings_dialog(page):
        return False
    for sel in [SEL_PERSONAL_TAB, 'text="Personalization"']:
        try:
            personal_btn = page.locator(sel).first
            if await personal_btn.is_visible(timeout=2000):
                await personal_btn.click()
                await page.wait_for_timeout(700)
                return True
        except Exception:
            pass
    return False


async def _clear_chatgpt_memory(page: Page, retries: int = 3) -> bool:
    """
    Open ChatGPT Settings → Personalization → Manage memories → delete all.
    Falls back to deleting memories one by one if no bulk-delete button exists.
    Returns True on success, False if all attempts failed.
    """
    for attempt in range(1, retries + 1):
        try:
            opened = await _open_settings_personalization(page)
            if not opened:
                raise RuntimeError("could not open Personalization settings")

            # Click the "Manage" button next to the Memory heading
            manage_btn = page.locator(SEL_MEMORY_MANAGE_BTN).first
            await manage_btn.click(timeout=5000)
            await page.wait_for_timeout(800)

            # Wait for the Saved memories popover to appear. The exact heading
            # tag is not stable, so avoid hard-coding `h2`.
            await page.wait_for_selector('text="Saved memories"', timeout=5000)
            await page.wait_for_timeout(400)

            # Prefer the concrete Saved memories overlay when present.
            modal_root = page.locator("#modal-golden-hour-memories")
            dialog_candidates = page.locator('[role="dialog"]').filter(
                has=page.locator('text="Saved memories"')
            )

            async def _debug_locator(locator, label: str, limit: int = 5) -> None:
                try:
                    count = await locator.count()
                    rows = []
                    for idx in range(min(count, limit)):
                        el = locator.nth(idx)
                        rows.append({
                            "text": (await el.inner_text()).strip()[:120] if await el.count() else "",
                            "aria": await el.get_attribute("aria-label") or "",
                            "testid": await el.get_attribute("data-testid") or "",
                            "role": await el.get_attribute("role") or "",
                        })
                    print(f"[debug] {label}: count={count} sample={rows}")
                except Exception as exc:
                    print(f"[debug] {label}: <error collecting debug: {exc}>")

            kebab = None
            try:
                if await modal_root.is_visible(timeout=1500):
                    print("[debug] Saved memories modal root is visible")
                    modal_kebabs = modal_root.locator(SEL_MEMORY_KEBAB)
                    await _debug_locator(modal_kebabs, "modal More options")
                    if await modal_kebabs.count() > 0:
                        kebab = modal_kebabs.first
            except Exception:
                pass

            if kebab is None:
                dialog_root = dialog_candidates.last
                try:
                    await dialog_root.wait_for(state="visible", timeout=2000)
                    print("[debug] Saved memories dialog candidate is visible")
                    dialog_kebabs = dialog_root.locator(SEL_MEMORY_KEBAB)
                    await _debug_locator(dialog_kebabs, "dialog More options")
                    if await dialog_kebabs.count() > 0:
                        kebab = dialog_kebabs.first
                except Exception:
                    pass

            if kebab is None:
                page_kebabs = page.locator(SEL_MEMORY_KEBAB)
                await _debug_locator(page_kebabs, "page More options")
                if await page_kebabs.count() > 0:
                    kebab = page_kebabs.last

            if kebab is None:
                raise RuntimeError("Saved memories opened, but no visible More options button was found")

            try:
                kebab_count = await page.locator(SEL_MEMORY_KEBAB).count()
                print(f"[debug] visible-ish More options candidates: {kebab_count}")
            except Exception:
                pass

            print("[debug] clicking Saved memories More options...")
            await kebab.wait_for(state="attached", timeout=2000)
            await kebab.click(force=True)
            await page.wait_for_selector('[role="menu"][data-state="open"]', timeout=3000)
            await page.wait_for_timeout(300)
            await _debug_locator(page.locator('[role="menu"][data-state="open"]'), "open menus")
            await _debug_locator(page.locator('[role="menu"][data-state="open"] [role="menuitem"]'), "open menu items")

            named_bulk = page.get_by_role("menuitem", name="Delete all memories")
            selector_bulk = page.locator(SEL_MEMORY_DELETE_ALL)
            await _debug_locator(named_bulk, "named delete-all memory menuitems")
            await _debug_locator(selector_bulk, "selector delete-all memory menuitems")

            bulk_btn = None
            if await named_bulk.count() > 0:
                bulk_btn = named_bulk.first
                print("[debug] using named Delete all memories menuitem")
            elif await selector_bulk.count() > 0:
                bulk_btn = selector_bulk.first
                print("[debug] using selector Delete all memories menuitem")

            if bulk_btn is not None:
                print("[debug] clicking Delete all memories...")
                await bulk_btn.wait_for(state="attached", timeout=2000)
                await bulk_btn.click(force=True)
                await page.wait_for_timeout(500)
                confirm = page.locator(SEL_CONFIRM_DELETE_MEMORIES).first
                await _debug_locator(page.locator(SEL_CONFIRM_DELETE_MEMORIES), "delete-all memory confirm buttons")
                if not await confirm.is_visible(timeout=3000):
                    raise RuntimeError("Delete-all-memories confirm button did not appear")
                print("[debug] confirming Delete all memories...")
                await confirm.click(force=True)
                await page.wait_for_timeout(800)
            else:
                try:
                    visible_menu_items = await page.evaluate(
                        """
                        () => [...document.querySelectorAll('[role="menuitem"]')]
                          .map(el => (el.innerText || el.textContent || '').trim())
                          .filter(Boolean)
                        """
                    )
                    print(f"[debug] visible menu items: {visible_menu_items}")
                except Exception:
                    pass
                raise RuntimeError("Saved memories menu opened, but 'Delete all memories' was not found")

            # Close nested overlays and return to the main chat UI.
            for _ in range(4):
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(300)
                except Exception:
                    pass
            try:
                await page.wait_for_selector(f"{SEL_INPUT}, {SEL_PROFILE_BTN}", timeout=5000)
            except Exception:
                pass
            return True

        except Exception as e:
            try:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)
                await page.keyboard.press("Escape")
            except Exception:
                pass
            if attempt < retries:
                print(f"[memory clear failed (attempt {attempt}/{retries}): {e}] retrying in 10s...")
                await asyncio.sleep(10)
            else:
                print(f"[memory clear failed after {retries} attempts: {e}]")
    return False


async def _send_message(
    page: Page,
    text: str,
    pre_send_delay: float = 0.0,
    post_response_delay: float = 8.0,
) -> None:
    input_box = page.locator(SEL_INPUT)
    await input_box.wait_for(state="visible", timeout=20_000)
    await input_box.click()
    await page.evaluate(
        """([sel, txt]) => {
            const el = document.querySelector(sel);
            if (!el) return;
            el.focus();
            if (el.isContentEditable) {
                el.innerText = txt;
                el.dispatchEvent(new Event('input', {bubbles: true}));
            } else {
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(el, txt);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }
        }""",
        [SEL_INPUT, text],
    )
    await page.wait_for_timeout(int(max(0.4, pre_send_delay) * 1000))
    send_btn = page.locator(SEL_SEND_BTN)
    await send_btn.wait_for(state="visible", timeout=10_000)
    await send_btn.click()
    await _wait_for_response(page)
    await page.wait_for_timeout(int(post_response_delay * 1000))


async def _wait_for_response(page: Page, timeout: int = 240_000) -> None:
    deadline = time.time() + timeout / 1000
    last_text = ""
    last_change_at = time.time()
    saw_assistant_message = False
    try:
        await page.wait_for_selector(SEL_STOP_BTN, timeout=15_000)
    except PWTimeout:
        pass
    while time.time() < deadline:
        current_text = ""
        try:
            msgs = page.locator(SEL_ASST_MSG)
            count = await msgs.count()
            if count > 0:
                saw_assistant_message = True
                current_text = (await msgs.nth(count - 1).inner_text()).strip()
        except Exception:
            current_text = ""

        now = time.time()
        if current_text != last_text:
            last_text = current_text
            last_change_at = now

        try:
            stop_visible = await page.locator(SEL_STOP_BTN).is_visible(timeout=500)
        except Exception:
            stop_visible = False

        # Preferred completion signal: streaming button disappears.
        if not stop_visible:
            # If an assistant message already appeared and then stopped changing,
            # treat the response as complete even if the streaming button state is flaky.
            if not saw_assistant_message or (now - last_change_at) >= 2.0:
                return

        # Fallback: if the last assistant message has appeared and stayed stable
        # for a while, accept it as complete even if the stop button never clears.
        if saw_assistant_message and (now - last_change_at) >= 5.0:
            return

        await page.wait_for_timeout(1000)
    raise TimeoutError("ChatGPT response timed out")


async def _get_last_response(page: Page) -> str:
    msgs = page.locator(SEL_ASST_MSG)
    count = await msgs.count()
    if count == 0:
        return ""
    return (await msgs.nth(count - 1).inner_text()).strip()


async def _check_memory_update(page: Page) -> tuple[bool, str]:
    """
    Detect if ChatGPT's Personalization memory feature was triggered.
    Looks for the memory update UI indicator that appears near the last message.
    Returns (triggered: bool, content: str describing what was saved).
    """
    try:
        # ChatGPT shows a memory chip/anchor near the message when memory is saved.
        # Try common selectors — update if UI changes.
        # The memory chip is a button[aria-haspopup="dialog"] inside div.inline-block,
        # appearing near the last assistant message after a memory save.
        mem_el = page.locator(
            'div.inline-block > button[aria-haspopup="dialog"]'
        ).last
        if await mem_el.is_visible(timeout=800):
            content = (await mem_el.inner_text()).strip()
            # Only count it if the text looks memory-related (avoids other dialog buttons)
            if any(kw in content.lower() for kw in ("memory", "saved", "updated", "remember")):
                return True, content
    except Exception:
        pass
    return False, ""


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

async def _run_session(
    page: Page,
    session: TestSession,
    timing: Optional[TimingProfile],
    turn_delay: float,
    history_rate: float,
    sample_id: str,
    world: str,
    conv_source: str,
) -> SessionRunResult:
    """
    Execute one TestSession: for each phase, feed user turns then ask MCQs.
    Returns records, a full trace, and human-readable log lines.
    """
    await _new_chat(page)
    results: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    log_lines: list[str] = []
    last_resp_len = 0
    session_memory_events: list[dict] = []

    def _log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    for phase in session.phases:
        # --- Feed user turns ---
        for j, turn in enumerate(phase.user_turns):
            pre  = timing.sample_typing_delay(len(turn)) if timing else 0.4
            post = (timing.sample_reading_delay(last_resp_len) * history_rate
                    if timing else turn_delay)
            prefix = (f"    [{phase.label}] turn {j+1}/{len(phase.user_turns)} "
                      f"({len(turn)}ch pre={pre:.1f}s post={post:.1f}s)...")
            try:
                await _send_message(page, turn, pre_send_delay=pre, post_response_delay=post)
                resp_text = await _get_last_response(page)
                last_resp_len = len(resp_text)
                mem_triggered, mem_content = await _check_memory_update(page)
                if mem_triggered:
                    event = {"phase": phase.label, "turn_index": j,
                             "user_turn": turn[:200], "memory_content": mem_content}
                    session_memory_events.append(event)
                    suffix = f" ok [MEMORY: {mem_content[:60]}]"
                else:
                    suffix = " ok"
                _log(prefix + suffix)
                trace.append({
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
                trace.append({
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

        # --- Ask MCQ questions ---
        for k, item in enumerate(phase.mcq_items):
            prompt = _build_mcq_prompt(item)
            pre  = timing.sample_typing_delay(len(prompt)) if timing else 0.4
            post = timing.sample_reading_delay(last_resp_len) if timing else turn_delay

            prefix = (f"    MCQ {k+1}/{len(phase.mcq_items)} "
                      f"[{item.qa_family} {item.turn_role} {item.timestamp}] "
                      f"pre={pre:.1f}s...")

            rec: dict[str, Any] = {
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
                await _send_message(page, prompt,
                                    pre_send_delay=pre, post_response_delay=post)
                response_text = await _get_last_response(page)
                last_resp_len = len(response_text)

                predicted = _extract_choice(response_text, item.choice_order)
                predicted_type = item.choice_to_answer_type.get(predicted, "")
                correct_type = item.choice_to_answer_type.get(item.correct_choice, "")
                rec["model_response"] = response_text
                rec["predicted_choice"] = predicted
                rec["predicted_answer_type"] = predicted_type
                mem_triggered, mem_content = await _check_memory_update(page)
                if mem_triggered:
                    event = {"phase": phase.label, "turn_index": f"mcq_{k}",
                             "user_turn": prompt[:200], "memory_content": mem_content}
                    session_memory_events.append(event)
                    rec["session_memory_events"] = list(session_memory_events)

                symbol = "✓" if predicted == item.correct_choice else "✗"
                mem_tag = " [MEMORY]" if mem_triggered else ""
                _log(
                    prefix + " " +
                    f"{symbol} pred={predicted} [pred_type={predicted_type}] "
                    f"correct={item.correct_choice} [correct_type={correct_type}]"
                    f"{mem_tag}"
                )
                trace.append({
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
                trace.append({
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()) or "unknown"


def _results_root(output_path: Path) -> Path:
    return output_path.parent / "chatgpt_web_results"


def _persona_output_dir(output_path: Path, topic: str, world: str, sample_id: str) -> Path:
    root = _results_root(output_path)
    return root / _slug(topic) / _slug(sample_id) / f"test_type_{_slug(world)}"


def _persona_output_path(output_path: Path, topic: str, world: str, sample_id: str) -> Path:
    return _persona_output_dir(output_path, topic, world, sample_id) / "results.jsonl"


def _session_output_dir(output_path: Path, topic: str, world: str, sample_id: str, session_key: str) -> Path:
    return _persona_output_dir(output_path, topic, world, sample_id) / f"session_{_slug(session_key)}"


def _session_artifact_paths(output_path: Path, topic: str, world: str, sample_id: str, session_key: str) -> dict[str, Path]:
    session_dir = _session_output_dir(output_path, topic, world, sample_id, session_key)
    return {
        "dir": session_dir,
        "result": session_dir / "session_result.json",
        "trace": session_dir / "session_trace.json",
        "log": session_dir / "session_log.txt",
    }


def _session_is_completed(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed"


def _write_session_artifacts(
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
    artifacts = _session_artifact_paths(output_path, topic, world, sample_id, session.session_key)
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
        "log_path": str(artifacts["log"]),
        "records": session_result.records,
    }
    artifacts["result"].write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["trace"].write_text(json.dumps(session_result.trace, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["log"].write_text("\n".join(session_result.log_lines) + "\n", encoding="utf-8")


async def login_only(args: argparse.Namespace) -> None:
    """
    Login-only mode: launch browser WITHOUT suppressing --disable-blink-features
    so Cloudflare verification passes. Wait for the user to log in, then exit.
    The session is saved to args.session_dir for use by the main evaluation.
    """
    print(f"Login-only mode. Session will be saved to: {args.session_dir}")
    async with async_playwright() as pw:
        context: BrowserContext = await pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.session_dir).resolve()),
            headless=False,
            channel="chrome",
            args=["--window-size=1280,900"],
            ignore_default_args=["--enable-automation", "--no-sandbox"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await context.add_init_script(
            "try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}"
        )
        await page.goto(CHATGPT_URL)
        print("Complete login and any Cloudflare verification in the browser window.")
        print("Press Enter here when you are fully logged in and see the chat page...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "")
        print("Session saved. You can now run the main evaluation.")
        await context.close()


async def _prompt_manual_cleanup(page: Page, label: str) -> None:
    loop = asyncio.get_event_loop()
    print(
        f"{label}: manually delete memory and delete the current chat in the browser, "
        "return to the main chat page, then press Enter to continue..."
    )
    await loop.run_in_executor(None, input, "")
    try:
        await page.goto(CHATGPT_URL)
        await page.wait_for_selector(SEL_INPUT, timeout=60_000)
        await page.wait_for_timeout(1500)
    except Exception:
        print("WARNING: chat input did not reappear after manual cleanup. Check the browser state.")


async def evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Resume: rely on per-session artifacts as the source of truth.
    done: set[tuple[str, str, str]] = set()
    persona_output_paths: list[Path] = []

    all_samples = _discover_samples(args.data_dir, args.topic)
    if args.sample_id_filter:
        all_samples = [s for s in all_samples if s.startswith(args.sample_id_filter)]
    if args.limit:
        all_samples = all_samples[: args.limit]

    print(f"Topic={args.topic}  World={args.world}  Personas={len(all_samples)}")

    timing: Optional[TimingProfile] = None
    if args.timing_profile and Path(args.timing_profile).exists():
        timing = TimingProfile.load(args.timing_profile)
        print(f"Using human timing profile ({timing.n_interactions} interactions)")
    elif args.timing_profile:
        print("WARNING: timing profile not found — using fixed delays")

    async with async_playwright() as pw:
        context: BrowserContext = await pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.session_dir).resolve()),
            headless=args.headless,
            channel="chrome",
            args=["--window-size=1280,900"],
            ignore_default_args=["--enable-automation", "--no-sandbox",
                                   "--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        # Patch navigator.webdriver before any page load
        await context.add_init_script(
            "try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}"
        )

        await page.goto(CHATGPT_URL)
        await page.wait_for_timeout(2000)

        # Wait for full login (only #prompt-textarea exists when ChatGPT is ready)
        print("If you see a Cloudflare or login page, complete it now (up to 5 minutes).")
        logged_in = False
        try:
            await page.wait_for_selector(SEL_INPUT, timeout=300_000)
            logged_in = True
            print("Ready.")
        except PWTimeout:
            print("WARNING: ChatGPT did not load — check the browser window.")

        # After the input appears, give the page a short grace period to settle.
        if logged_in:
            print(f"Waiting {args.ready_grace_sec:.1f}s before starting cleanup...")
            await page.wait_for_timeout(int(max(0.0, args.ready_grace_sec) * 1000))
            if args.manual_cleanup:
                print("Pre-run cleanup: manual mode")
                await _prompt_manual_cleanup(page, "Pre-run cleanup")
            else:
                print("Pre-run cleanup: clearing memory and deleting any existing chat...")
                await _clear_chatgpt_memory(page)
                await _delete_current_chat(page)
        else:
            print("Skipping pre-run cleanup (not logged in).")

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
                persona_output_path = _persona_output_path(output_root, args.topic, args.world, sample_id)
                persona_output_path.parent.mkdir(parents=True, exist_ok=True)
                persona_output_paths.append(persona_output_path)
                print(f"  {len(sessions)} session(s) planned: "
                      f"{[s.session_key for s in sessions]}")

                for sess in sessions:
                    triple = (sample_id, args.world, sess.session_key)
                    artifacts = _session_artifact_paths(output_root, args.topic, args.world, sample_id, sess.session_key)
                    if not args.overwrite and _session_is_completed(artifacts["result"]):
                        print(f"  [{sess.session_key}] already done, skipping")
                        done.add(triple)
                        continue

                    phase_summary = " → ".join(
                        f"{p.label}({len(p.user_turns)}turns"
                        + (f"+{len(p.mcq_items)}MCQs" if p.mcq_items else "") + ")"
                        for p in sess.phases
                    )
                    print(f"  [{sess.session_key}] {phase_summary}")

                    try:
                        session_result = await _run_session(
                            page, sess, timing, args.turn_delay,
                            args.history_rate,
                            sample_id, args.world, conv_source,
                        )
                        session_error = ""
                        session_status = "completed"
                    except Exception as e:
                        print(f"  SESSION ERROR: {e}")
                        session_result = SessionRunResult(
                            records=[{
                                "sample_id": sample_id,
                                "world": args.world,
                                "session_key": sess.session_key,
                                "error": str(e),
                            }],
                            trace=[],
                            log_lines=[f"SESSION ERROR: {e}"],
                        )
                        session_error = str(e)
                        session_status = "error"
                        try:
                            await page.goto(CHATGPT_URL)
                            await page.wait_for_timeout(3000)
                        except Exception:
                            pass

                    # Persist per-session artifacts before touching aggregate output.
                    _write_session_artifacts(
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

                    # Write persona-level aggregate results atomically (all at once per session)
                    with open(str(persona_output_path), "a", encoding="utf-8") as out_f:
                        for rec in session_result.records:
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(triple)

                    # Cleanup before the next session.
                    if args.manual_cleanup:
                        await _prompt_manual_cleanup(page, f"  [{sess.session_key}] manual cleanup")
                    else:
                        print("  Clearing ChatGPT memory...", end=" ", flush=True)
                        ok = await _clear_chatgpt_memory(page)
                        print("ok" if ok else "FAILED — please clear manually before next session")
                        print("  Deleting chat...", end=" ", flush=True)
                        ok2 = await _delete_current_chat(page)
                        print("ok" if ok2 else "FAILED — please delete manually before next session")

                    delay = timing.sample_reading_delay(200) if timing else args.sample_delay
                    await page.wait_for_timeout(int(delay * 1000))

        finally:
            await context.close()

    unique_outputs = []
    seen = set()
    for path in persona_output_paths:
        if path not in seen:
            unique_outputs.append(path)
            seen.add(path)

    print(f"\nDone. Persona-level results saved under {_results_root(output_root)}")
    for path in unique_outputs:
        print(f"  {path}")
        _print_summary(path)


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
    print(f"\n=== Summary ===")
    print(f"Total: {total}  Errors: {errors}  Answered: {len(answered)}")
    if answered:
        print(f"Overall correct: {correct}/{len(answered)} ({100*correct/len(answered):.1f}%)")
    family_totals: dict[str, int] = {}
    family_errors: dict[str, int] = {}
    family_answered: dict[str, dict[str, int]] = {}
    for r in records:
        fam = r.get("qa_family", "?")
        family_totals[fam] = family_totals.get(fam, 0) + 1
        if r.get("error") or not r.get("predicted_choice"):
            family_errors[fam] = family_errors.get(fam, 0) + 1
            continue
        t = r.get("predicted_answer_type", "unknown")
        family_answered.setdefault(fam, {})
        family_answered[fam][t] = family_answered[fam].get(t, 0) + 1
    for fam in sorted(family_totals):
        counts = family_answered.get(fam, {})
        answered_n = sum(counts.values())
        error_n = family_errors.get(fam, 0)
        correct_n = counts.get("remember_correct", 0)
        print(
            f"  {fam}: correct {correct_n}/{answered_n} | "
            f"answered {answered_n}/{family_totals[fam]} | errors {error_n} | {counts}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChatGPT web on PersonaMem recall MCQs via Playwright"
    )
    parser.add_argument("--topic", default="travelPlanning")
    parser.add_argument("--world", default="baseline",
                        choices=["baseline", "forget", "no_store"],
                        help="Memory-control world (no_use: TODO)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default="results/chatgpt_web_baseline.jsonl")
    parser.add_argument("--timing_profile", default="",
                        help="Path to human_timing.json")
    parser.add_argument("--turn_delay", type=float, default=8.0,
                        help="Fixed post-response delay in seconds (no timing profile)")
    parser.add_argument("--history_rate", type=float, default=0.2,
                        help="Multiplier on reading delay for history turns (default 0.3 = 3x faster)")
    parser.add_argument("--sample_delay", type=float, default=5.0,
                        help="Fixed inter-session delay in seconds (no timing profile)")
    parser.add_argument("--session_dir", default="./chatgpt_session")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--login", action="store_true",
                        help="Login-only mode: open browser, wait for manual login, then exit. "
                             "Run this once before the main evaluation to save the session.")
    parser.add_argument("--manual_cleanup", action="store_true",
                        help="Before pre-run cleanup and before each session cleanup, pause and let "
                             "the user manually delete memory and delete the current chat.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N personas (0 = all)")
    parser.add_argument("--sample_id_filter", default="",
                        help="Only process sample_ids with this prefix")
    parser.add_argument("--ready_grace_sec", type=float, default=5.0,
                        help="After ChatGPT input becomes visible, wait this many seconds before cleanup/evaluation starts.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.login:
        asyncio.run(login_only(args))
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
