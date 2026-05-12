"""MemTree (Memory-in-the-LLM-Era) adapter.

The vendored MemTree adapter
constructs the GlobalConfig-shaped state in-process (skipping the original
config.yaml + Milvus auto-init), routes every internal LLM call to OpenRouter,
and shims out the multiprocessing pool that the official `modify_nodes` step
uses (forked workers don't see our monkey-patches reliably).

Write phase per turn (user-only — see NOTE):
  * embed turn (sentence-transformer, CPU on this machine)
  * descend the tree by depth-aware cosine threshold (Milvus Lite per sample)
  * insert new leaf under the chosen parent
  * AGGREGATE_PROMPT every parent on the traversal path → re-embed + upsert
  * re-attach the parent's pre-update text as a sibling leaf

Read phase per MCQ:
  * embed question → milvus top-K cosine over ALL node vectors
  * resolve each hit through `tree.nodes[id].cv` (which may be a current
    AGGREGATE summary rather than the raw turn)
  * concat with \\n\\n into the shared memory-aware MCQ prompt; answer via request_text

NOTE (CLAUDE.md rule 4 – assistant turns):
    MemTree's per-add_node cost scales linearly (1–3 AGGREGATE LLM calls per
    parent on the traversal path). Feeding the assistant turn would double
    write-phase cost without adding test signal — assistant utterances are
    never the subject of MCQ probes. So we ingest only `role == "user"` turns.
    Mirrors the user-only ingestion convention used by retrieval backends with
    per-turn write cost.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from ..base import MethodAdapter
from ..utils import load_official_memtree_module
from ...shared import (
    build_memory_eval_prompt,
    ensure_openai_env,
    load_openai_client,
    request_text,
    resolve_model_name,
)


# ---------------------------------------------------------------------------
# Embedding-model dimensions for the popular default + a few common picks. The
# Milvus collection has to be created with a matching `dimension`, so we infer
# it from the model name. Override with the method config `embedding_model` if
# you pass an exotic model.
# ---------------------------------------------------------------------------

_EMBEDDING_DIMS: Dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "bge-m3": 1024,
    "BAAI/bge-m3": 1024,
    "models--BAAI--bge-m3": 1024,
}


def _safe_token(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return cleaned.strip("_") or "default"


def _resolve_embedding_dim(model_name: str) -> int:
    if model_name in _EMBEDDING_DIMS:
        return _EMBEDDING_DIMS[model_name]
    # Last-ditch: try probing the model.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        dim = int(model.get_sentence_embedding_dimension())
        _EMBEDDING_DIMS[model_name] = dim
        return dim
    except Exception:
        return 384


class _SerialPool:
    """Drop-in for `multiprocessing.Pool` that runs tasks serially in-process.

    MemTree's `modify_nodes` uses `multiprocessing.Pool(...).imap_unordered(...)`
    to parallelize AGGREGATE calls. Forked workers don't reliably inherit our
    monkey-patched `worker_ollama` / `globalconfig`, so we serialize.
    """

    def __init__(self, processes: int = 1, **_: Any) -> None:
        del processes

    def __enter__(self) -> "_SerialPool":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def imap_unordered(self, func: Any, iterable: Any) -> Any:
        for item in iterable:
            yield func(item)

    def close(self) -> None:
        return None

    def join(self) -> None:
        return None


class _SerialMP:
    """Replacement for `multiprocessing` in the structure module's namespace."""

    Pool = _SerialPool


def _new_token_log() -> Dict[str, Dict[str, int]]:
    return {
        bucket: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "failed": 0}
        for bucket in ("aggregate", "answer")
    }


def _record_usage(usage_log: Dict[str, int], usage: Any) -> None:
    """Pull token counts off an OpenAI/OpenRouter response.usage payload."""
    if usage is None:
        return
    pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
    ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
    tt = getattr(usage, "total_tokens", None) or (pt + ct)
    usage_log["prompt_tokens"] += int(pt or 0)
    usage_log["completion_tokens"] += int(ct or 0)
    usage_log["total_tokens"] += int(tt or 0)


def _build_counting_chat_call(client: Any, resolved_model: str, usage_log: Dict[str, int]) -> Any:
    """OpenRouter chat.completions.create wrapper that records token usage.

    Returned callable takes a fully-formed messages list and returns the
    assistant content. Token counts go into ``usage_log`` (mutated in place)
    so the adapter can report cumulative cost per phase.
    """

    def _call(messages: List[Dict[str, str]], *, temperature: float = 0.0) -> str:
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            usage_log["calls"] += 1
            usage_log["failed"] += 1
            print(f"[memtree] OpenRouter call failed: {exc}", flush=True)
            return ""
        usage_log["calls"] += 1
        _record_usage(usage_log, getattr(resp, "usage", None))
        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return ""
        return (content or "").strip()

    return _call


def _build_openrouter_worker(call_with_usage: Any) -> Any:
    """Replacement for vendored ``worker_ollama`` / ``worker_openai``.

    Wraps the token-counting chat.completions caller into the original
    ``worker_ollama(prompt) -> str`` signature so ``modify_nodes`` and
    ``_modify_shared_mem`` work unchanged.
    """

    def _worker(prompt: str) -> str:
        return call_with_usage([{"role": "user", "content": prompt}], temperature=0.0)

    return _worker


def _build_cpu_get_embedding(model: Any) -> Any:
    """CPU-safe replacement for vendored `get_embedding` (which hard-codes
    `device='cuda'`). The host doesn't always have a usable CUDA driver."""
    import numpy as np

    device = "cuda" if _has_usable_cuda() else "cpu"

    def _get_embedding(texts: Any, batch: int = 1) -> Any:
        embs = model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
            batch_size=batch,
        )
        embs = embs.cpu().numpy()
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs

    return _get_embedding


def _has_usable_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class MemTreeAdapter(MethodAdapter):
    backend_name = "memtree_retrieval"

    def __init__(
        self,
        *,
        memtree_module: Any,
        client: Any,
        model: str,
        resolved_model: str,
        config_ns: SimpleNamespace,
        persona_messages: List[Dict[str, str]],
        memory_limit: int,
        token_log: Dict[str, Dict[str, int]],
        answer_call: Any,
    ) -> None:
        self.memtree_module = memtree_module
        self.client = client
        self.model = model
        self.resolved_model = resolved_model
        self.config_ns = config_ns
        self.persona_messages = persona_messages
        self.memory_limit = memory_limit
        self.token_log = token_log
        self.answer_call = answer_call
        self.tree = memtree_module.structure.MemTree("")
        self.root_id = id(self.tree.root)
        self.preload_log: Dict[str, Any] = {
            "input_messages": [],
            "preload_steps": [],
            "written_nodes": [],   # per-user-turn nodes added to the tree
            "tree_size": 1,
            "config_ns": {
                "embedding_model_name": getattr(config_ns, "embedding_model_name", ""),
                "dimension": getattr(config_ns, "dimension", 0),
                "base_threshold": getattr(config_ns, "base_threshold", None),
                "rate": getattr(config_ns, "rate", None),
                "max_depth": getattr(config_ns, "max_depth", None),
                "top_k_retrieve": getattr(config_ns, "top_k_retrieve", None),
                "collection_name": getattr(config_ns, "collection_name", ""),
                "db_name": getattr(config_ns, "db_name", ""),
            },
        }

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        del ask_period
        if stage_batches:
            for batch in stage_batches:
                self._preload_one_stage(batch["messages"], stage_label=batch.get("period", ""))
        else:
            self._preload_one_stage(context_messages, stage_label="")

    def _preload_one_stage(self, messages: List[Dict[str, str]], stage_label: str = "") -> None:
        # User-only ingestion (CLAUDE.md rule 4 — see module docstring).
        user_turns = [m for m in messages if m.get("role") == "user" and m.get("content", "").strip()]
        if not user_turns:
            return
        step_log: Dict[str, Any] = {
            "stage": stage_label,
            "user_turn_count": len(user_turns),
            "tree_size_before": self.tree.size,
        }
        for idx, msg in enumerate(user_turns):
            content = msg["content"].strip()
            self.preload_log["input_messages"].append({"role": "user", "content": content})
            tree_size_before_add = self.tree.size
            self.tree.add_node(content, self.root_id)
            # Per-turn write log: record what we asked MemTree to ingest +
            # whether it grew the tree (was kept) vs. merged into an existing
            # node (no growth). Lets the report show exactly which user turns
            # made it into the store.
            self.preload_log["written_nodes"].append({
                "stage": stage_label,
                "content": content,
                "tree_size_before": tree_size_before_add,
                "tree_size_after": self.tree.size,
                "added_new_node": self.tree.size > tree_size_before_add,
            })
            if idx == 0 or idx == len(user_turns) - 1 or (idx + 1) % 5 == 0:
                print(
                    f"[memtree] preload {stage_label or 'stage'}: {idx + 1}/{len(user_turns)} user turns "
                    f"(tree size: {self.tree.size})",
                    flush=True,
                )
        step_log["tree_size_after"] = self.tree.size
        # Dump every node's CURRENT cv. Internal nodes' cv was rewritten by
        # AGGREGATE_PROMPT during the turn-by-turn ingest above; this lets the
        # report show exactly what summaries MemTree settled on. Leaves still
        # carry the original turn text (plus, per modify_nodes, the parent's
        # pre-AGGREGATE text gets re-attached as a sibling leaf, which is the
        # main reason MemTree "can't forget" — both abstract summary and raw
        # detail end up retrievable).
        step_log["store_snapshot"] = self._snapshot_tree_nodes()
        self.preload_log["preload_steps"].append(step_log)
        self.preload_log["tree_size"] = self.tree.size
        self.preload_log["store_snapshot"] = step_log["store_snapshot"]

    def _snapshot_tree_nodes(self) -> List[Dict[str, Any]]:
        """Dump (id, depth, parent_id, child_count, cv) for every node. Sorted
        by depth ascending so root → internal-summary → leaves reads naturally
        in the report. Each cv is the *current* text — for internal nodes
        that's the latest AGGREGATE summary; for leaves it's the original
        turn (or the re-attached pre-AGGREGATE text from modify_nodes)."""
        nodes = self.tree.nodes
        adjacency = self.tree.adjacency
        out: List[Dict[str, Any]] = []
        for node_id, node in nodes.items():
            out.append({
                "id": node_id,
                "depth": getattr(node, "dv", 0),
                "parent_id": getattr(node, "pv", None),
                "child_count": len(adjacency.get(node_id, set()) or set()),
                "cv": str(getattr(node, "cv", "")),
            })
        out.sort(key=lambda r: (r["depth"], r["id"]))
        return out

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        get_embedding = self.memtree_module.utils.get_embedding
        search = self.memtree_module.utils.search

        query_emb = get_embedding(question, getattr(self.config_ns, "embedding_batch_size", 256))
        # Milvus client expects list[list[float]] — flatten to a 2D list.
        query_payload = [list(query_emb[0])]
        top_k = getattr(self.config_ns, "top_k_retrieve", self.memory_limit)
        try:
            hits = search(query_payload, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"[memtree] milvus search failed: {exc}", flush=True)
            hits = []

        # Search returns [[hit, hit, ...]] — unwrap.
        hit_list = hits[0] if hits and isinstance(hits, list) else []
        hit_ids: List[int] = []
        retrieved_texts: List[str] = []
        for hit in hit_list:
            try:
                node_id = hit["id"]
            except (KeyError, TypeError):
                continue
            node = self.tree.nodes.get(node_id)
            if node is None or not node.cv:
                continue
            hit_ids.append(node_id)
            retrieved_texts.append(str(node.cv))

        memories_text = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant memories were retrieved."
        messages = self.persona_messages + [
            {
                "role": "user",
                "content": build_memory_eval_prompt(question, choices, memories_text),
            }
        ]
        response = self.answer_call(messages, temperature=0.0)
        return {
            "model_response": response,
            "retrieved_memories": {
                "node_ids": hit_ids,
                "texts": retrieved_texts,
                "raw_milvus_hits": [
                    {"id": h.get("id"), "score": h.get("distance")} if isinstance(h, dict) else str(h)
                    for h in hit_list
                ],
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        agg = self.token_log["aggregate"]
        ans = self.token_log["answer"]
        total_prompt = agg["prompt_tokens"] + ans["prompt_tokens"]
        total_completion = agg["completion_tokens"] + ans["completion_tokens"]
        total_tokens = agg["total_tokens"] + ans["total_tokens"]
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "memtree_source": "vendored_memory_in_the_llm_era",
            "token_usage": {
                "model": self.resolved_model,
                "aggregate": dict(agg),
                "answer": dict(ans),
                "total": {
                    "calls": agg["calls"] + ans["calls"],
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_tokens,
                    "failed": agg["failed"] + ans["failed"],
                },
            },
        }


def _build_config_ns(
    *,
    args: Any,
    runtime_root: Path,
    embedding_model_name: str,
    dimension: int,
    embedding_model: Any,
    milvus_client: Any,
    collection_name: str,
    db_name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        # storage
        client=milvus_client,
        collection_name=collection_name,
        db_name=db_name,
        save_path=str(runtime_root / "memtree.pkl"),
        save_name="memtree.pkl",
        vdb_name="milvus.db",
        # embedding
        model=embedding_model,
        embedding_model_name=embedding_model_name,
        dimension=dimension,
        embedding_batch_size=getattr(args, "memtree_embedding_batch_size", 64),
        # tree-traversal hyperparameters (matches memtree's config/config.yaml defaults)
        base_threshold=getattr(args, "memtree_base_threshold", 0.4),
        rate=getattr(args, "memtree_rate", 0.5),
        max_depth=getattr(args, "memtree_max_depth", 15),
        top_k_retrieve=getattr(args, "memtree_top_k_retrieve", 10),
        # we serialize, so this is mostly informational
        llm_parallel_nums=1,
        # internal LLM (will be unused after our monkeypatch, but kept so any
        # stray reference inside the vendored code finds something coherent)
        llm_base_url=os.environ.get("OPENAI_BASE_URL", ""),
        llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
        llm_model=resolve_model_name(args.model),
        # dataset metadata
        dataset_name=Path(args.rendered).stem.replace(".recall_rendered", ""),
        dataset_path="",
    )


def _resolve_runtime_root(args: Any) -> Path:
    explicit = getattr(args, "memtree_runtime_root", "") or ""
    if explicit:
        return Path(explicit)
    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
    world = getattr(args, "world", "baseline")
    return Path("data/runtime/memtree") / world / stem


def _reset_runtime_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> MemTreeAdapter:
    ensure_openai_env(args.api_key_file)
    os.environ["MODEL"] = args.model
    embedding_model_name = getattr(args, "embedding_model", "") or "all-MiniLM-L6-v2"
    if embedding_model_name:
        os.environ["EMBEDDING_MODEL"] = embedding_model_name

    runtime_root = _resolve_runtime_root(args)
    if getattr(args, "memtree_reset_runtime", True):
        _reset_runtime_root(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    memtree_module = load_official_memtree_module()
    client = load_openai_client(args.api_key_file)
    resolved_model = resolve_model_name(args.model)

    # Sentence-transformer (defaults to all-MiniLM-L6-v2 → dim 384, CPU-safe).
    from sentence_transformers import SentenceTransformer  # type: ignore

    device = "cuda" if _has_usable_cuda() else "cpu"
    sbert = SentenceTransformer(embedding_model_name, device=device)
    dimension = _resolve_embedding_dim(embedding_model_name)

    # Per-(world, persona) Milvus Lite store. Filename is unique to this
    # eval run so multiple methods can coexist on disk.
    stem = Path(args.rendered).stem.replace(".recall_rendered", "")
    world = _safe_token(getattr(args, "world", "baseline"))
    db_name = str(runtime_root / "milvus.db")
    from pymilvus import MilvusClient  # type: ignore

    milvus_client = MilvusClient(db_name)
    collection_name = f"memtree_{_safe_token(world)}_{_safe_token(stem)}"
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )

    config_ns = _build_config_ns(
        args=args,
        runtime_root=runtime_root,
        embedding_model_name=embedding_model_name,
        dimension=dimension,
        embedding_model=sbert,
        milvus_client=milvus_client,
        collection_name=collection_name,
        db_name=db_name,
    )

    # Patch the singleton globalconfig everywhere it gets imported with
    # `from .config import globalconfig` (rebinds the module-level name in
    # each importer — must be done BEFORE we touch any function that reads it).
    for submod_name in ("config", "utils", "structure", "dataloader"):
        submod = getattr(memtree_module, submod_name, None)
        if submod is not None:
            setattr(submod, "globalconfig", config_ns)

    # Route the internal AGGREGATE_PROMPT call through OpenRouter, with a
    # token-counting wrapper so we can report cost in debug_payload.
    token_log = _new_token_log()
    aggregate_call = _build_counting_chat_call(client, resolved_model, token_log["aggregate"])
    answer_call = _build_counting_chat_call(client, resolved_model, token_log["answer"])
    openrouter_worker = _build_openrouter_worker(aggregate_call)
    memtree_module.utils.worker_ollama = openrouter_worker
    memtree_module.utils.worker_openai = openrouter_worker
    memtree_module.structure.worker_ollama = openrouter_worker
    memtree_module.structure.worker_openai = openrouter_worker

    # CPU-safe get_embedding (vendored impl pins device='cuda').
    cpu_get_embedding = _build_cpu_get_embedding(sbert)
    memtree_module.utils.get_embedding = cpu_get_embedding
    memtree_module.structure.get_embedding = cpu_get_embedding

    # Serialize the AGGREGATE pool. Forked workers wouldn't see the patches
    # above; since llm_parallel_nums=1 in our config, the pool would only
    # spin up one child anyway — but keep the in-process invariant strict.
    memtree_module.structure.multiprocessing = _SerialMP

    return MemTreeAdapter(
        memtree_module=memtree_module,
        client=client,
        model=args.model,
        resolved_model=resolved_model,
        config_ns=config_ns,
        persona_messages=persona_messages,
        memory_limit=getattr(args, "memory_limit", 5),
        token_log=token_log,
        answer_call=answer_call,
    )
