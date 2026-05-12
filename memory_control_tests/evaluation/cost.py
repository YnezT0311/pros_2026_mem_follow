from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CostSnapshot:
    usage: Optional[float]
    error: str = ""


def _load_openrouter_key(api_key_file: str) -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    path = Path(api_key_file)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def snapshot_openrouter_usage(api_key_file: str) -> CostSnapshot:
    key = _load_openrouter_key(api_key_file)
    if not key:
        return CostSnapshot(None, "missing OpenRouter key")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return CostSnapshot(None, f"{type(exc).__name__}: {exc}")
    try:
        return CostSnapshot(float(payload.get("data", {}).get("usage", 0.0)))
    except (TypeError, ValueError):
        return CostSnapshot(None, "usage field is not numeric")


def log_cost_delta(start: CostSnapshot, end: CostSnapshot) -> None:
    if start.usage is None or end.usage is None:
        reason = end.error or start.error or "usage unavailable"
        print(f"[cost] OpenRouter usage unavailable: {reason}", file=sys.stderr)
        return
    delta = end.usage - start.usage
    print(
        f"[cost] OpenRouter usage start=${start.usage:.4f} "
        f"end=${end.usage:.4f} delta=${delta:.4f}",
        file=sys.stderr,
    )
