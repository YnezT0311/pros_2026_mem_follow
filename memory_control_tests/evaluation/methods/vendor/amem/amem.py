"""Vendored A-Mem wrapper modeled on `A-mem/test_advanced_robust.py`.

A-Mem ships its memory layer (`memory_layer_robust.py`) outside this package.
We first try a normal import, then optionally inject `AMEM_REPO_ROOT` onto
`sys.path` when the caller explicitly points at a clone. We do not fall back to
machine-specific sibling paths.

The keyword-extraction step at `search_memory` follows test_advanced_robust.py
lines 96-107: a short LLM call turns a question into comma-separated keywords,
which then feed `find_related_memories_raw`.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from openai import OpenAI


REQUEST_TIMEOUT_SEC = 90
MAX_REQUEST_ATTEMPTS = 5
RETRYABLE_ERROR_MARKERS = (
    "connection error",
    "timed out",
    "timeout",
    "rate limit",
    "server error",
    "502",
    "503",
    "504",
)


def _resolve_amem_repo_root() -> Path:
    repo_root = os.getenv("AMEM_REPO_ROOT", "").strip()
    if not repo_root:
        raise FileNotFoundError(
            "Could not import A-Mem modules. Install A-Mem on PYTHONPATH or set AMEM_REPO_ROOT."
        )
    candidate = Path(repo_root)
    if not candidate.exists():
        raise FileNotFoundError(
            f"A-Mem repo not found at AMEM_REPO_ROOT={candidate}."
        )
    return candidate


def _ensure_amem_on_path() -> None:
    repo_root = str(_resolve_amem_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _run_with_retries(fn, description: str):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_REQUEST_ATTEMPTS or not _is_retryable(exc):
                raise
            sleep_s = min(30, 2 ** (attempt - 1))
            print(
                f"[A-Mem retry] {description} failed on attempt {attempt}/{MAX_REQUEST_ATTEMPTS}: "
                f"{type(exc).__name__}: {exc}. Retrying in {sleep_s}s.",
                flush=True,
            )
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{description} failed without raising an exception")


def _patch_openai_controller_for_openrouter(controller_cls: Any) -> None:
    """Make A-Mem's RobustOpenAIController route through OpenRouter.

    A-Mem's stock controller talks to api.openai.com and forces `max_tokens`,
    which OpenRouter's gpt-5* slugs reject (they require `max_completion_tokens`).
    The patch (a) injects the OpenRouter base URL and Title header, and
    (b) chooses the right max-token parameter based on the model slug.
    """
    if getattr(controller_cls, "_memoryctrl_openrouter_patch_applied", False):
        return

    title = os.getenv("OPENROUTER_TITLE", "MemoryCtrl")

    def _patched_init(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-OpenRouter-Title": title},
        )

    def _patched_get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "timeout": REQUEST_TIMEOUT_SEC,
        }
        if "gpt-5" in str(self.model):
            request_kwargs["max_completion_tokens"] = 1000
        else:
            request_kwargs["max_tokens"] = 1000
        response = _run_with_retries(
            lambda: self.client.chat.completions.create(**request_kwargs),
            "A-Mem RobustOpenAIController chat.completions.create",
        )
        return response.choices[0].message.content or ""

    controller_cls.__init__ = _patched_init
    controller_cls.get_completion = _patched_get_completion
    controller_cls._memoryctrl_openrouter_patch_applied = True


def _import_amem_modules() -> Tuple[Any, Any, Any]:
    try:
        from memory_layer_robust import (
            RobustAgenticMemorySystem,
            RobustLLMController,
            RobustOpenAIController,
        )
    except ImportError:
        _ensure_amem_on_path()
        try:
            from memory_layer_robust import (
                RobustAgenticMemorySystem,
                RobustLLMController,
                RobustOpenAIController,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import A-Mem robust modules. Install A-Mem on PYTHONPATH "
                "or set AMEM_REPO_ROOT to a clone containing memory_layer_robust.py."
            ) from exc
    return RobustAgenticMemorySystem, RobustLLMController, RobustOpenAIController


class AMem:
    """Thin wrapper exposing add_memory / search_memory.

    Mirrors the surface used by `RobustAdvancedMemAgent` in
    `A-mem/test_advanced_robust.py`: `add_note` is the write path,
    `find_related_memories_raw` plus an upstream keyword-generation LLM
    call is the read path.
    """

    def __init__(
        self,
        *,
        model: str,
        embedding_model: str,
        api_key: Optional[str] = None,
    ) -> None:
        RobustAgenticMemorySystem, RobustLLMController, RobustOpenAIController = _import_amem_modules()
        _patch_openai_controller_for_openrouter(RobustOpenAIController)
        self._RobustLLMController = RobustLLMController
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self.memory_system = RobustAgenticMemorySystem(
            model_name=embedding_model,
            llm_backend="openai",
            llm_model=model,
            api_key=self._api_key,
        )
        self._retriever_llm = RobustLLMController(
            backend="openai",
            model=model,
            api_key=self._api_key,
        )

    def reset(self) -> None:
        self.memory_system.memories = {}
        retriever = getattr(self.memory_system, "retriever", None)
        if retriever is None:
            return
        if hasattr(retriever, "corpus"):
            retriever.corpus = []
        if hasattr(retriever, "embeddings"):
            retriever.embeddings = None
        if hasattr(retriever, "document_ids"):
            retriever.document_ids = {}

    def add_memory(self, content: str, time: Optional[str] = None) -> str:
        return self.memory_system.add_note(content=content, time=time)

    def generate_query_keywords(self, question: str) -> str:
        prompt = (
            f"Given the following question, generate several keywords separated by commas.\n\n"
            f"Question: {question}\n\nKeywords:"
        )
        return self._retriever_llm.llm.get_completion(prompt, temperature=0.0)

    def search_memory(self, query: str, k: int = 10) -> str:
        return self.memory_system.find_related_memories_raw(query, k=k)
