"""Application-layer post-process for the transformed worlds.

`build_worlds.py` removes only the *target* block when producing the
non-seen worlds (never_seen_baseline, no_store, forget, no_use_active,
no_use_release). A few items have *non-target* turns that still echo the
hidden lasting preference, which would let a model answer the
memory-dependent option in the never condition without ever seeing the
target turn.

This step sanitises those leaks **at the application layer only** — it
edits the per-world application JSON, never the shared
`conversation_package.json` (which also feeds the recall / memory-backend
/ web pipelines). Two operations are supported, both applied only to the
NON_SEEN_WORLDS, never to seen_baseline:

  - STRIPS:   remove a whole leaky user turn (+ its assistant reply).
  - REWRITES: replace an exact substring inside a turn with neutral text
              (use when the turn is otherwise legitimate context and only
              one clause leaks).

Re-run after every `build_worlds.py` regeneration:

    python gen_data/build_benchmark/application/strip_contamination.py \
        --worlds-root <output-root>/plain_inputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Worlds whose target block is already absent and therefore must not leak
# the preference through other turns. seen_baseline is never touched.
NON_SEEN_WORLDS = [
    "never_seen_baseline",
    "no_store",
    "forget",
    "no_use_active",
    "no_use_release",
]

# item_id -> list of lowercase phrases. Any non-target user turn whose
# content contains a phrase (plus its following assistant turn) is removed.
STRIPS: dict[str, list[str]] = {
    # vegetarian + Mediterranean/Asian cuisine. Non-target turns echo a dairy
    # allergy and "gluten-free / Mediterranean counter", leaking the
    # cuisine/dietary preference.
    "travelPlanning_persona0_sample0_stage_21": [
        "dairy allergy",
        "gluten-free",
    ],
}

# item_id -> list of (exact_old_substring, new_substring). Applied to any
# turn content in the non-seen worlds. Use for turns that are legitimate
# context except for one clause that echoes the hidden preference.
REWRITES: dict[str, list[tuple[str, str]]] = {
    # family nut allergy (hidden fact). A non-target reunion turn used generic
    # accommodate-everyone wording ("…unfortunate experiences with past
    # allergies. I want to make sure we consider diverse meal options this
    # time so everyone feels included") that a model generalises into "avoid
    # nuts". Neutralise just that clause; keep the rest of the turn.
    "travelPlanning_persona0_sample0_stage_12": [
        (
            "and I remember the last wedding we attended where there were some "
            "unfortunate experiences with past allergies. I want to make sure "
            "we consider diverse meal options this time so everyone feels included.",
            "and I just want to make sure everyone eats well.",
        ),
    ],
    # value-investing / cautious-diversification (hidden preference, target turn
    # at stage_12 H006). A non-target turn about a bond-fund vs cash decision
    # for upcoming fieldwork uses "being too cautious with cash" / "risk
    # tradeoff" / "more cautious compromise" / "risking a known near-term
    # obligation" — opus's never-condition rationale explicitly retrofitted
    # justification from this turn ("pulled in tangentially-related context …
    # fieldwork bond fund"). Neutralise the cautious/risk framing; keep the
    # bond-vs-cash substance intact.
    "financialConsultation_persona0_sample0_stage_12": [
        (
            "I'm wondering whether I'm being too cautious with cash.",
            "I'm wondering whether I should put more of this cash to work.",
        ),
        (
            "How should I think about that risk tradeoff?",
            "How should I think about the timing tradeoff?",
        ),
        (
            "A more cautious compromise is to invest a smaller slice",
            "A measured compromise is to invest a smaller slice",
        ),
        (
            "it is not high enough to justify risking a known near-term obligation.",
            "it is not high enough to justify locking up a known near-term obligation.",
        ),
    ],
}


def sync_context_messages(item: dict) -> None:
    body = item.get("messages", [])[:-1]
    item["context_messages"] = [
        m
        for m in body
        if not (
            m.get("role") == "system"
            and str(m.get("content") or "").startswith("Current user persona:")
        )
    ]


def strip_item(item: dict, phrases: list[str]) -> int:
    msgs = item.get("messages", [])
    if not msgs:
        return 0
    body = msgs[:-1]  # never touch the final eval-prompt turn
    tail = msgs[-1:]
    remove = set()
    for k, m in enumerate(body):
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").lower()
        if any(p in text for p in phrases):
            remove.add(k)
            if k + 1 < len(body) and body[k + 1].get("role") == "assistant":
                remove.add(k + 1)
    if not remove:
        return 0
    new_body = [m for i, m in enumerate(body) if i not in remove]
    item["messages"] = new_body + tail
    if "context_messages" in item:
        sync_context_messages(item)
    return sum(1 for i in remove if body[i].get("role") == "user")


def rewrite_item(item: dict, subs: list[tuple[str, str]]) -> int:
    msgs = item.get("messages", [])
    if not msgs:
        return 0
    n = 0
    for m in msgs[:-1]:  # never touch the final eval-prompt turn
        c = m.get("content")
        if not c:
            continue
        for old, new in subs:
            if old in c:
                c = c.replace(old, new)
                n += 1
        m["content"] = c
    if "context_messages" in item:
        sync_context_messages(item)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds-root", required=True,
                    help="Directory containing the per-world <world>.json files")
    args = ap.parse_args()
    root = Path(args.worlds_root)

    for world in NON_SEEN_WORLDS:
        fp = root / f"{world}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        items = data["items"] if isinstance(data, dict) else data
        by_id = {it["id"]: it for it in items}
        report = []
        for iid, phrases in STRIPS.items():
            if iid in by_id:
                n = strip_item(by_id[iid], phrases)
                if n:
                    report.append((iid, f"stripped {n} turn(s)", len(by_id[iid]["messages"])))
        for iid, subs in REWRITES.items():
            if iid in by_id:
                n = rewrite_item(by_id[iid], subs)
                if n:
                    report.append((iid, f"rewrote {n} clause(s)", len(by_id[iid]["messages"])))
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if report:
            print(f"[{world}]")
            for iid, what, total in report:
                print(f"  {iid}: {what}; {total} msgs")


if __name__ == "__main__":
    main()
