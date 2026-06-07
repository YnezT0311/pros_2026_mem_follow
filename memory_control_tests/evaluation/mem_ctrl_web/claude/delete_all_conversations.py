#!/usr/bin/env python3
"""
Standalone "delete ALL Claude.ai conversations" helper.

Why this exists
---------------
The eval runner calls `_delete_all_chat_history` between sessions, but it:
  * caps at max_deletions=50 (too few when conversations have piled up), and
  * breaks the whole loop on the FIRST error,
so in practice it often deletes 0 and chats accumulate.

This script drives the same UI deletion but in an OUTER loop: it keeps invoking
the batch deleter until the sidebar is empty (or a hard cap), so a single
transient error or a >50 backlog no longer stops it. It also prints a clear
diagnostic (how many sidebar rows it can see) so if 0 get deleted you can tell
whether the list is genuinely empty or the selectors are stale.

Uses the SAME persistent browser session as the evaluator (`--session_dir`,
default ./claude_session) — log in once with
`python evaluate_claude_web.py --login` first.

Usage
-----
    python delete_all_conversations.py                 # delete everything
    python delete_all_conversations.py --max-total 3000
    python delete_all_conversations.py --headless
"""
import argparse
import asyncio
from pathlib import Path

from patchright.async_api import BrowserContext, async_playwright

# Reuse the evaluator's browser helpers (same directory).
import evaluate_claude_web as ev


async def _count_rows(page) -> int:
    try:
        return await page.locator(ev.SEL_CHAT_ROW).count()
    except Exception:
        return -1


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
            log(f"WARNING: Claude not confirmed ready ({exc}). Make sure you are logged in:\n"
                f"  python evaluate_claude_web.py --login --session_dir {args.session_dir}")

        # Diagnostic: how many conversation rows can we see?
        await ev._ensure_sidebar_expanded(page, debug_log=(log if args.debug else None))
        await page.wait_for_timeout(1000)
        seen = await _count_rows(page)
        log(f"\nVisible sidebar conversation rows (selector {ev.SEL_CHAT_ROW!r}): {seen}")
        if seen == 0:
            log("Nothing visible to delete. If you KNOW there are conversations, the sidebar "
                "selector is likely stale (claude.ai changed) — re-run with --debug and send me "
                "the output so I can fix the selectors.")
            if not args.headless:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, input, "Press Enter to close...")
            await context.close()
            return

        # Outer loop: keep deleting batches until empty / no progress / cap.
        total = 0
        stalls = 0
        while total < args.max_total:
            deleted, err = await ev._delete_all_chat_history(
                page,
                max_deletions=min(args.batch, args.max_total - total),
                debug_log=(log if args.debug else None),
            )
            total += deleted
            remaining = await _count_rows(page)
            log(f"  batch: deleted={deleted}  total={total}  remaining≈{remaining}"
                + (f"  last_error={err}" if err else ""))
            if remaining == 0:
                break
            if deleted == 0:
                stalls += 1
                if stalls == 1 and err:
                    log("  (no progress this batch — retrying once after a short wait)")
                    await page.wait_for_timeout(2000)
                    await page.goto(ev.CLAUDE_URL)
                    await ev._ensure_sidebar_expanded(page, debug_log=(log if args.debug else None))
                    continue
                log("  STALLED: rows are present but none could be deleted — the menu/delete "
                    "selectors are likely stale. Re-run with --debug and send me the snapshot.")
                break
            stalls = 0

        final = await _count_rows(page)
        log(f"\nDone. Deleted {total} conversation(s). Remaining visible: {final}.")
        if final and final > 0:
            log("Some conversations remain — re-run the script (it resumes), or delete the "
                "stragglers manually, or send me --debug output if it's stuck at 0 progress.")
        if not args.headless:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input, "Press Enter to close the browser...")
        await context.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Delete ALL Claude.ai conversations (standalone).")
    p.add_argument("--session_dir", default="./claude_session",
                   help="Persistent browser profile (same one the evaluator uses).")
    p.add_argument("--batch", type=int, default=200,
                   help="Deletions attempted per inner batch before re-counting.")
    p.add_argument("--max-total", type=int, default=5000,
                   help="Safety cap on total deletions.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Print selector snapshots (use this if nothing deletes).")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
