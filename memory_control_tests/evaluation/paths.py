from __future__ import annotations

from pathlib import Path

from ..common import period_tag


METHOD_FILENAME_TAG = {
    "plain": "raw_eval",
    "mem0": "mem0_retrieval_eval",
    "amem": "a_mem_retrieval_eval",
    "langmem": "langmem_retrieval_eval",
    "memoryos": "memoryos_retrieval_eval",
    "memtree": "memtree_retrieval_eval",
}

METHOD_ADAPTER_DIR_TAG = {
    "plain": "",
    "mem0": "mem0",
    "amem": "A-Mem",
    "langmem": "LangMem",
    "memoryos": "MemoryOS",
    "memtree": "MemTree",
}


def model_tag_for_filename(method: str, model: str) -> str:
    if method == "langmem":
        return "".join(ch if ch.isalnum() else "_" for ch in str(model))
    return str(model).replace("/", "_")


def rendered_stem(rendered: str) -> str:
    path = Path(rendered)
    name = path.name
    if name == "mcq_questions.json":
        return path.parent.name
    for suffix in (".mcq_questions.json",):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def topic_from_rendered(rendered: str) -> str:
    parts = Path(rendered).parts
    if len(parts) >= 3 and parts[-2] == "specs":
        return parts[-3]
    if len(parts) >= 3 and parts[-1] == "mcq_questions.json":
        return parts[-3]
    raise ValueError(
        f"Cannot infer topic from rendered path {rendered!r}; expected "
        "'<...>/data/mcq_work/<topic>/<persona>/mcq_questions.json'."
    )


def default_output_path(
    rendered: str,
    world: str,
    ask_period: str,
    method: str,
    model: str,
    *,
    no_use_restrict_period: str = "",
    no_use_release_period: str = "",
) -> str:
    method_tag = METHOD_FILENAME_TAG.get(method, f"{method}_eval")
    model_tag = model_tag_for_filename(method, model)
    if world == "no_use":
        suffix = f".{world}.restrict_{period_tag(no_use_restrict_period or 'all_stages')}"
        if no_use_release_period:
            suffix += f".release_{period_tag(no_use_release_period)}"
        suffix += f".test_{period_tag(ask_period)}.{method_tag}_{model_tag}.json"
    else:
        suffix = f".{world}.{period_tag(ask_period)}.{method_tag}_{model_tag}.json"

    topic = topic_from_rendered(rendered)
    stem = rendered_stem(rendered)
    adapter_tag = METHOD_ADAPTER_DIR_TAG.get(method, method)
    model_dir = f"{str(model).replace('/', '_')}"
    folder = f"{model_dir}+{adapter_tag}" if adapter_tag else model_dir
    return str(Path("eval_results") / topic / world / folder / f"{stem}{suffix}")
