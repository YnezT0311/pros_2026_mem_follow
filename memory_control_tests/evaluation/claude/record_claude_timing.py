#!/usr/bin/env python3
"""
record_claude_timing.py

Interactive timing recorder for Claude Web evaluation.

This script helps regenerate `human_timing.json` by having you perform a few
natural Claude interactions in the browser while you mark timing checkpoints in
the terminal.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


SAMPLE_PROMPTS = [
    (
        "I want you to remember a few travel preferences because I often ask related questions later. "
        "I strongly prefer spontaneous travel over tightly scheduled itineraries, I dislike coordinating "
        "large group trips because they become chaotic, I prefer characterful local stays over resorts, "
        "and I usually value flexibility and local feel more than optimization or luxury."
    ),
    (
        "Please help me design a short trip-planning workflow that fits my style. I want something that "
        "lets me keep plans loose, compare a few neighborhoods without overcommitting, note any local "
        "cafes or photo spots I should keep in mind, and still leave room for last-minute changes if "
        "weather or mood shifts."
    ),
    (
        "I am working on travel planning and want to keep a clear memory of my preferences and questions "
        "so I can make better trips. Please keep in mind that I usually avoid big group travel, prefer "
        "small-party or solo travel, like dramatic landscapes and road trips, and care a lot about "
        "practical apps that make spontaneity easier."
    ),
    (
        "Please summarize what you know so far about my travel style in five bullet points, then add a "
        "short paragraph about what kinds of destinations, accommodations, and planning tools probably "
        "fit me best based on those preferences."
    ),
    (
        "I need a more detailed planning answer now. Build me a compact decision framework for choosing "
        "between three possible trips: a slow coastal road trip, a city-based photography weekend, and "
        "a mixed train-and-walk itinerary through smaller towns. Compare them on flexibility, stress, "
        "visual appeal, local character, and how compatible they are with spontaneous decision-making."
    ),
    (
        "Please remember one more preference: I like answers that are simple, direct, and clear rather "
        "than overly polished or long-winded. If you save that, tell me briefly what you stored and how "
        "it would affect the way you answer me later."
    ),
    (
        "Now answer this in a way that looks closer to an evaluation prompt: Based on everything I said "
        "earlier, which of these sounds most like my style: (a) highly structured group tours with fixed "
        "timelines, (b) last-minute solo or small-party trips with flexible plans and local character, "
        "or (c) luxury resort travel optimized months in advance? Reply with just the letter and one "
        "short sentence of explanation."
    ),
    (
        "Finally, imagine I ask you to clear memory at the end of a session. Tell me what you think you "
        "currently remember about my preferences, then state what would need to happen for that memory "
        "to be removed so the next session starts clean."
    ),
]


def _mean(values: list[float], fallback: float) -> float:
    return statistics.mean(values) if values else fallback


def _std(values: list[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    return statistics.stdev(values)


def _read_multiline_response() -> str:
    print("Paste Claude's response below. Finish with an empty line.")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a Claude-specific human_timing.json")
    parser.add_argument("--output", default="human_timing.json")
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    rounds = min(args.rounds, len(SAMPLE_PROMPTS))
    print("")
    print("Claude timing recorder")
    print("=====================")
    print("For each round:")
    print("1. Read the sample prompt here.")
    print("2. In Claude Web, type it naturally and send it.")
    print("3. Mark the checkpoints in this terminal when prompted.")
    print("4. These prompts are intentionally longer and closer to the real evaluation workload.")
    print("")

    typing_cps = []
    pre_send_pauses = []
    reading_cps = []
    post_reading_pauses = []

    for idx, prompt in enumerate(SAMPLE_PROMPTS[:rounds], start=1):
        print("")
        print(f"[Round {idx}/{rounds}]")
        print(f"Prompt ({len(prompt)} chars): {prompt}")
        input("Press Enter when you are ready to begin this round...")

        input("Press Enter the moment you START typing this prompt in Claude...")
        type_start = time.time()
        input("Press Enter immediately AFTER you click Send in Claude...")
        send_time = time.time()

        typing_duration = max(0.1, send_time - type_start)
        typing_cps.append(len(prompt) / typing_duration)

        pause_raw = input("Optional: how many seconds did you pause before sending after finishing typing? [default 1.5] ").strip()
        try:
            pre_send_pauses.append(float(pause_raw) if pause_raw else 1.5)
        except ValueError:
            pre_send_pauses.append(1.5)

        input("Press Enter the moment Claude FINISHES streaming its response...")
        response_done = time.time()
        input("Now read the response naturally. Press Enter AFTER you finish reading it...")
        read_done = time.time()

        read_duration = max(0.1, read_done - response_done)
        response_text = _read_multiline_response()
        if not response_text:
            fallback = input("If you do not want to paste the response, enter approximate character count: ").strip()
            response_len = int(fallback) if fallback.isdigit() else 200
        else:
            response_len = len(response_text)
        reading_cps.append(response_len / read_duration)

        think_raw = input("Optional: how many seconds did you pause after reading before acting again? [default 4.0] ").strip()
        try:
            post_reading_pauses.append(float(think_raw) if think_raw else 4.0)
        except ValueError:
            post_reading_pauses.append(4.0)

    payload = {
        "typing_chars_per_sec_mean": round(_mean(typing_cps, 42.0), 3),
        "typing_chars_per_sec_std": round(max(1.0, _std(typing_cps, 12.0)), 3),
        "pre_send_pause_mean": round(_mean(pre_send_pauses, 1.8), 3),
        "pre_send_pause_std": round(max(0.2, _std(pre_send_pauses, 0.7)), 3),
        "reading_chars_per_sec_mean": round(_mean(reading_cps, 18.0), 3),
        "reading_chars_per_sec_std": round(max(1.0, _std(reading_cps, 5.0)), 3),
        "post_reading_pause_mean": round(_mean(post_reading_pauses, 6.0), 3),
        "post_reading_pause_std": round(max(0.5, _std(post_reading_pauses, 1.5)), 3),
        "min_turn_delay": 8.0,
        "max_turn_delay": 180.0,
        "n_interactions": rounds,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("")
    print("Saved timing profile:")
    print(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
