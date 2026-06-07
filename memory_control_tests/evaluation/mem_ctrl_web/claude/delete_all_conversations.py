#!/usr/bin/env python3
"""
Standalone "delete ALL Claude.ai conversations" helper.

Why this exists
---------------
The eval runner's `_delete_all_chat_history` (a) caps at 50, (b) breaks on the
first error, and (c) clicks a confirm button by a hard-coded data-testid
(`delete-modal-confirm`) that goes stale when claude.ai changes its UI — so the
delete modal opens but the final "Delete" never gets clicked and chats pile up.

This script reuses the evaluator's persistent session + row/menu helpers but
implements its own deletion loop with a ROBUST confirm step (multiple
fallbacks: data-testid -> in-dialog "Delete" text -> Enter), runs in an outer
loop until the sidebar is empty, and survives transient errors.

Log in once first:
    python evaluate_claude_web.py --login

Usage
-----
    python delete_all_conversations.py            # delete everything
    python delete_all_conversations.py --debug    # verbose (use if it gets stuck)
    python delete_all_conversations.py --headless
"""
import argparse
import asyncio
from pathlib import Path

from patchright.async_api import BrowserContext, async_playwright

import evaluate_claude_web as ev  # same directory; reuse browser helpers + selectors

# Confirm-button fallbacks, tried in order. The modal is a role=dialog; the
# confirm action is a button labelled "Delete" inside it.
CONFIRM_SELECTORS = [
    ev.SEL_DELETE_CONFIRM,                                  # original data-testid
    '[role="dialog"] [data-testid="delete-modal-confirm"]',
    'div[role="dialog"] button:has-text("Delete")',
    '[role="dialog"] button:has-text("Delete")',
    'div[role="alertdialog"] button:has-text("Delete")',
    'button:has-text("Delete chat")',
    'button:has-text("Delete conversation")',
]


async def _count_rows(page) -> int:
    try:
        return await page.locator(ev.SEL_CHAT_ROW).count()
    except Exception:
        return -1


async def _click_confirm(page, log, debug: bool) -> bool:
    """Click the delete-confirm button using several fallbacks; finally try Enter."""
    for sel in CONFIRM_SELECTORS:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            await loc.wait_for(state="visible", timeout=1500)
            await loc.click()
            if debug:
                log(f"    confirmed via selector: {sel}")
            return True
        except Exception:
            continue
    # last resort: many modals confirm on Enter
    try:
        await page.keyboard.press("Enter")
        if debug:
            log("    confirmed via Enter key")
        return True
    except Exception:
        return False


async def _delete_one(page, log, debug: bool) -> bool:
    """Delete the top conversation row. Returns True if a deletion was performed."""
    rows = page.locator(ev.SEL_CHAT_ROW)
    if await rows.count() == 0:
        return False
    await ev._hover_chat_row(page, 0, debug_log=(log if debug else None))
    trigger = page.locator(ev.SEL_CHAT_ROW_MENU_TRIGGER).first
    await trigger.wait_for(state="visible", timeout=3000)
    await ev._focus_chat_row_for_menu(page, trigger)
    await trigger.click()
    await page.wait_for_timeout(300)

    delete_item = page.locator(ev.SEL_DELETE_CHAT_TRIGGER).first
    await delete_item.wait_for(state="visible", timeout=2500)
    await delete_item.click()
    await page.wait_for_timeout(400)

    if not await _click_confirm(page, log, debug):
        raise RuntimeError("delete-confirm button not found (modal open but not confirmed)")
    await page.wait_for_timeout(1000)
    return True


async def run(args: argparse.Namespace) -> None:
    log = print
    async with async_playwright() as pw:
        context: BrowserContext = await pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.session_dir).resolve()),
            headless=args.headless,
            args=[
                "--window-size=1280,900",
                "--disable-features=DnsOverHttps,EncryptedClientHello",
                "--disable-blink-features=AutomationControlled",
            ],
            ignore_default_args=["--enable-automation", "--no-sandbox"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await context.add_init_script(
            "try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}"
        )
        await page.goto(ev.CLAUDE_URL)
        try:
            await ev._wait_for_claude_ready(page, timeout=60_000)
        except Exception as exc:
            log(f"WARNING: Claude not confirmed ready ({exc}). Log in first:\n"
                f"  python evaluate_claude_web.py --login --session_dir {args.session_dir}")

        await ev._ensure_sidebar_expanded(page, debug_log=(log if args.debug else None))
        await page.wait_for_timeout(1000)
        seen = await _count_rows(page)
        log(f"\nVisible sidebar conversations ({ev.SEL_CHAT_ROW!r}): {seen}")
        if seen == 0:
            log("Nothing to delete (or the row selector is stale — try --debug).")
            await _pause_close(context, page, args)
            return

        total, stalls = 0, 0
        while total < args.max_total:
            try:
                did = await _delete_one(page, log, args.debug)
            except Exception as exc:
                did = False
                log(f"  delete error: {exc}")
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass

            if did:
                total += 1
                stalls = 0
                if total % 10 == 0 or args.debug:
                    log(f"  deleted {total} (remaining≈{await _count_rows(page)})")
                continue

            # no deletion this iteration
            remaining = await _count_rows(page)
            if remaining == 0:
                break
            stalls += 1
            if stalls <= 2:
                log(f"  no progress (remaining≈{remaining}); refreshing and retrying...")
                await page.goto(ev.CLAUDE_URL)
                await ev._ensure_sidebar_expanded(page, debug_log=(log if args.debug else None))
                await page.wait_for_timeout(1500)
                continue
            log("  STALLED: rows present but cannot delete. Re-run with --debug and send me "
                "the output (which confirm selector, if any, matched).")
            break

        log(f"\nDone. Deleted {total} conversation(s). Remaining visible: {await _count_rows(page)}.")
        await _pause_close(context, page, args)


async def _pause_close(context, page, args) -> None:
    if not args.headless:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "Press Enter to close the browser...")
    await context.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Delete ALL Claude.ai conversations (standalone).")
    p.add_argument("--session_dir", default="./claude_session",
                   help="Persistent browser profile (same one the evaluator uses).")
    p.add_argument("--max-total", type=int, default=5000, help="Safety cap on total deletions.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Verbose: print which confirm selector matched / row snapshots.")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
