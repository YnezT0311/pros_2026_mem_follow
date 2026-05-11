#!/usr/bin/env python3
"""
record_claude_web_clicks.py

Standalone helper for recording manual clicks in the Claude web UI.
Use this to capture the actual selectors/buttons from your account after login.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from patchright.async_api import BrowserContext, async_playwright

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evaluate_claude_web import CLAUDE_URL


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
        await page.goto(CLAUDE_URL)
        await page.wait_for_timeout(1500)
        await page.evaluate(RECORDER_SCRIPT)

        print("Manually click the Claude controls you want to inspect, then press Enter here.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "")

        clicks = await page.evaluate("() => window.__codex_click_log || []")
        output_path.write_text(json.dumps(clicks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(clicks)} click events to {output_path}")
        await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record manual Claude web clicks to JSON")
    parser.add_argument("--session_dir", default="./claude_session")
    parser.add_argument("--output", default="results/claude_clicks.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
