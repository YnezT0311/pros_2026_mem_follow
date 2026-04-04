#!/usr/bin/env python3
"""
record_chatgpt_web_clicks.py

Standalone helper for recording manual clicks in the ChatGPT web UI.
This is intentionally separate from `evaluate_chatgpt_web.py` so the
main evaluator can stay stable while selector debugging stays isolated.

Typical usage:
  python record_chatgpt_web_clicks.py \
    --session_dir ./chatgpt_session \
    --output results/chatgpt_clicks.json \
    --open_saved_memories
"""

import argparse
import asyncio
import json
from pathlib import Path

from patchright.async_api import async_playwright, BrowserContext

from evaluate_chatgpt_web import (
    CHATGPT_URL,
    SEL_MEMORY_MANAGE_BTN,
    _open_settings_personalization,
)


RECORDER_SCRIPT = r"""
() => {
  window.__codex_click_log = [];
  function cssPath(el) {
    if (!el || !(el instanceof Element)) return "";
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === Node.ELEMENT_NODE && parts.length < 10) {
      let part = cur.tagName.toLowerCase();
      if (cur.id) {
        part += "#" + cur.id;
        parts.unshift(part);
        break;
      }
      const role = cur.getAttribute("role");
      const dt = cur.getAttribute("data-testid");
      const aria = cur.getAttribute("aria-label");
      if (dt) part += `[data-testid="${dt}"]`;
      if (role) part += `[role="${role}"]`;
      if (aria) part += `[aria-label="${aria}"]`;
      if (!dt && !role && !aria && cur.classList.length) {
        part += "." + [...cur.classList].slice(0, 2).join(".");
      }
      const parent = cur.parentElement;
      if (parent) {
        const siblings = [...parent.children].filter(x => x.tagName === cur.tagName);
        if (siblings.length > 1) {
          part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
        }
      }
      parts.unshift(part);
      cur = parent;
    }
    return parts.join(" > ");
  }
  document.addEventListener("click", (ev) => {
    const el = ev.target instanceof Element ? ev.target.closest("*") : null;
    if (!el) return;
    window.__codex_click_log.push({
      ts: new Date().toISOString(),
      url: location.href,
      tag: el.tagName.toLowerCase(),
      text: (el.innerText || el.textContent || "").trim().slice(0, 200),
      aria_label: el.getAttribute("aria-label") || "",
      data_testid: el.getAttribute("data-testid") || "",
      role: el.getAttribute("role") || "",
      classes: [...el.classList].slice(0, 10),
      css_path: cssPath(el),
    });
  }, true);
}
"""


async def main_async(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        await page.wait_for_timeout(1500)
        await page.evaluate(RECORDER_SCRIPT)

        if args.open_saved_memories:
            try:
                if await _open_settings_personalization(page):
                    manage_btn = page.locator(SEL_MEMORY_MANAGE_BTN).first
                    if await manage_btn.is_visible(timeout=3000):
                        await manage_btn.click(timeout=5000)
                        await page.wait_for_selector('text="Saved memories"', timeout=5000)
                        await page.wait_for_timeout(500)
                        print("Recorder opened the Saved memories UI.")
                    else:
                        print("Recorder could not find the Manage memories button.")
                else:
                    print("Recorder could not open Personalization settings.")
            except Exception as exc:
                print(f"Recorder could not auto-open Saved memories: {exc}")

        print("Manually click the controls you want to inspect, then press Enter here.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "")

        clicks = await page.evaluate("() => window.__codex_click_log || []")
        output_path.write_text(json.dumps(clicks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(clicks)} click events to {output_path}")
        await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record manual ChatGPT web clicks to JSON")
    parser.add_argument("--session_dir", default="./chatgpt_session")
    parser.add_argument("--output", default="results/chatgpt_clicks.json")
    parser.add_argument("--open_saved_memories", action="store_true",
                        help="Try to open Settings -> Personalization -> Saved memories before recording.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
