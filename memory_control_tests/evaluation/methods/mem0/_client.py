"""Self-hosted mem0 `Memory` instantiation for the benchmark.

The mem0 paper's evaluation uses the managed `MemoryClient`. This benchmark
runs mem0 self-hosted (local Qdrant + history db) so it stays comparable to
the other self-hosted memory backends (A-Mem, LangMem) under the same
OpenRouter model. See `_patches.py` for the validation layer that compensates
for what the managed service does internally.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ._patches import (
    MEM0_STRICT_UPDATE_MEMORY_PROMPT,
    disable_history_writes,
    install_update_action_guard,
)


def ensure_mem0_home(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp_root = root / "tmp"
    qdrant_root = root / "qdrant"
    tmp_root.mkdir(parents=True, exist_ok=True)
    qdrant_root.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(root)
    os.environ["MEM0_DIR"] = str(root)
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)


def reset_runtime_root(runtime_root: Path) -> None:
    if runtime_root.exists():
        shutil.rmtree(runtime_root)


def load_local_mem0_memory(runtime_root: Path, *, llm_model: str) -> Any:
    """Build a self-hosted `Memory` and install the mem0 patches.

    Caller is expected to have already populated `OPENAI_API_KEY` /
    `OPENAI_BASE_URL` (via `shared.ensure_openai_env`).
    """
    ensure_mem0_home(runtime_root)

    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    qdrant_path = runtime_root / "qdrant"
    history_db_path = runtime_root / "history.db"
    config = MemoryConfig(
        history_db_path=str(history_db_path),
        custom_update_memory_prompt=MEM0_STRICT_UPDATE_MEMORY_PROMPT,
        llm={
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
                "openrouter_base_url": os.getenv("OPENAI_BASE_URL", ""),
                "app_name": "MemoryCtrl",
            },
        },
        vector_store={
            "provider": "qdrant",
            "config": {
                "path": str(qdrant_path),
            },
        },
    )
    memory = Memory(config=config)
    install_update_action_guard(memory)
    disable_history_writes(memory)
    return memory
