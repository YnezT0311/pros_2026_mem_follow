from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


METHODS = ("plain", "mem0", "langmem", "amem", "memoryos", "memtree")
WORLDS = ("baseline", "no_store", "forget")


METHOD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "plain": {},
    "mem0": {
        "memory_limit": 10,
        "embedding_model": "",
        "preload_batch_size": 2,
        "mem0_runtime_root": "",
        "mem0_reset_runtime": True,
    },
    "langmem": {
        "memory_limit": 10,
        "embedding_model": "text-embedding-3-small",
        "preload_batch_size": 2,
    },
    "amem": {
        "memory_limit": 5,
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "memoryos": {
        "embedding_model": "all-MiniLM-L6-v2",
        "memoryos_runtime_root": "",
        "memoryos_reset_runtime": True,
        "memoryos_short_term_capacity": 10,
        "memoryos_mid_term_capacity": 2000,
        "memoryos_long_term_knowledge_capacity": 100,
        "memoryos_retrieval_queue_capacity": 7,
    },
    "memtree": {
        "memory_limit": 5,
        "embedding_model": "all-MiniLM-L6-v2",
        "memtree_runtime_root": "",
        "memtree_reset_runtime": True,
        "memtree_base_threshold": 0.4,
        "memtree_rate": 0.5,
        "memtree_max_depth": 15,
        "memtree_top_k_retrieve": 10,
        "memtree_embedding_batch_size": 64,
    },
}


PERIOD_TAG_TO_FULL: Dict[str, str] = {
    "initial": "Conversation Initial Stage",
    "early": "Conversation Early Stage",
    "intermediate": "Conversation Intermediate Stage",
    "late": "Conversation Late Stage",
}


def _resolve_period_list(spec: str) -> list[str]:
    """Parse `--ask_periods` value (e.g. 'early,intermediate,late') into the
    canonical PERIODS list. Empty string returns []. Unknown tags raise.
    """
    spec = spec.strip()
    if not spec:
        return []
    out: list[str] = []
    for raw in spec.split(","):
        tag = raw.strip().lower()
        if not tag:
            continue
        if tag not in PERIOD_TAG_TO_FULL:
            raise ValueError(
                f"Unknown ask_period tag '{tag}'. Use one of: {', '.join(PERIOD_TAG_TO_FULL)}"
            )
        out.append(PERIOD_TAG_TO_FULL[tag])
    return out


@dataclass
class EvalConfig:
    rendered: str
    method: str
    model: str = "gpt-oss-120b"
    ask_period: str = "all_stages"
    # When non-empty, run all listed periods in one process with a single
    # incremental preload. Each period writes its own output file.
    # Falls back to single `ask_period` when empty.
    # Retired for stage-N data; mem_evals rejects it.
    ask_periods: list[str] = field(default_factory=list)
    world: str = "baseline"
    sidecar: str = ""
    output: str = ""
    api_key_file: str = "keys/openrouter_key.txt"
    reasoning_effort: str = ""
    workers: int = 10
    method_config: Dict[str, Any] = field(default_factory=dict)

    def adapter_args(self) -> SimpleNamespace:
        """Expose one namespace for existing method adapters.

        The public CLI only has common evaluation settings. Backend-specific
        settings live in `method_config`, then get flattened here for the
        adapter factories.
        """
        values = {
            "rendered": self.rendered,
            "method": self.method,
            "model": self.model,
            "ask_period": self.ask_period,
            "world": self.world,
            "sidecar": self.sidecar,
            "output": self.output,
            "api_key_file": self.api_key_file,
            "reasoning_effort": self.reasoning_effort,
            "workers": self.workers,
        }
        values.update(self.method_config)
        return SimpleNamespace(**values)


def load_method_config(method: str, path: str = "") -> Dict[str, Any]:
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"Unsupported method: {method}")
    config = dict(METHOD_DEFAULTS[method])
    if not path:
        return config

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Method config must be a JSON object: {path}")
    unknown = sorted(set(loaded) - set(config))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown config key(s) for method {method}: {joined}")
    config.update(loaded)
    return config


def parse_eval_config() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Run a MemoryCtrl evaluation. Backend-specific settings go in --method_config JSON."
    )
    parser.add_argument("--rendered", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--method_config", default="", help="JSON object with settings for the selected method.")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--ask_period", default="all_stages")
    parser.add_argument(
        "--ask_periods",
        default="",
        help=(
            "Deprecated. Stage-N data is evaluated once at all_stages; this option "
            "is rejected by memory_control_tests.evaluation.mem_evals."
        ),
    )
    parser.add_argument("--world", choices=WORLDS, default="baseline")
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--api_key_file", default="keys/openrouter_key.txt")
    parser.add_argument(
        "--reasoning_effort",
        default="",
        help="Forward `reasoning: {effort: ...}` to OpenRouter when set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max parallel MCQ workers. Only used by stateless adapters.",
    )
    args = parser.parse_args()
    return EvalConfig(
        rendered=args.rendered,
        method=args.method,
        model=args.model,
        ask_period=args.ask_period,
        ask_periods=_resolve_period_list(args.ask_periods),
        world=args.world,
        sidecar=args.sidecar,
        output=args.output,
        api_key_file=args.api_key_file,
        reasoning_effort=args.reasoning_effort,
        workers=args.workers,
        method_config=load_method_config(args.method, args.method_config),
    )
