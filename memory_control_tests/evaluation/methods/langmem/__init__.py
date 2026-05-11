import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import MethodAdapter
from ..utils import load_official_langmem_module
from ...shared import build_eval_prompt, ensure_openai_env, load_openai_client, request_text


OFFICIAL_STYLE_MCq_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from one user's conversation history. These memories may contain information that is relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze the provided memories.
2. If the question asks about a specific fact, look for direct evidence in the memories.
3. If the memories contain contradictory information, prioritize the most recent or most specific evidence.
4. If there is not enough evidence to support a confident remembered answer, choose the option that appropriately says the information is not remembered.
5. Focus only on the content of the memories and the question.
6. Answer the multiple-choice question by selecting exactly one option label.

Memories:

{memories}

Question:
{question}

Options:
{options}

Return only the final answer label after the special token <final_answer>, for example <final_answer>(b)</final_answer>.
"""


def _render_choice_lines(choices: Dict[str, str]) -> str:
    return "\n".join(f"({label.lower()}) {text}" for label, text in choices.items())


def _format_store_hits(items: List[Any]) -> str:
    """Render store.search results as plain text for the answer prompt."""
    lines = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            text = str(item)
            if text.strip():
                lines.append(f"{idx}. {text}")
            continue
        value = item.get("value")
        content = ""
        if isinstance(value, dict):
            content = str(value.get("content", "")).strip()
        elif isinstance(value, str):
            content = value.strip()
        if not content:
            content = str(value) if value is not None else ""
        if content:
            lines.append(f"{idx}. {content}")
    return "\n".join(lines)


def _safe_namespace_token(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return cleaned.strip("_") or "default_user"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    return repr(value)


class LangMemAdapter(MethodAdapter):
    backend_name = "langmem_retrieval"

    def __init__(
        self,
        *,
        langmem_impl: Any,
        client: Any,
        model: str,
        user_id: str,
        memory_limit: int,
        preload_batch_size: int,
    ) -> None:
        self.langmem_impl = langmem_impl
        self.client = client
        self.model = model
        self.user_id = user_id
        self.memory_limit = memory_limit
        self.preload_batch_size = max(1, preload_batch_size)
        self.runtime_user_id = _safe_namespace_token(user_id)
        self.config = {"configurable": {"thread_id": f"langmem-{self.runtime_user_id}"}}
        self.preload_log: Dict[str, Any] = {
            "input_messages": [],
            "manager_writes": [],
            "store_snapshot": [],
            "runtime_user_id": self.runtime_user_id,
        }
        # Build a search-only react agent that shares the same store as the
        # preload agent. This is used at answer time so the LLM can do
        # query rewriting + multi-step retrieval (LangMem's intended usage)
        # while being physically incapable of writing the MCQ question into
        # the memory store (no manage_memory_tool in its tool list).
        self.query_agent = self._build_query_agent()
        self._query_counter = 0

    def _build_query_agent(self) -> Any:
        """search-only react agent sharing the underlying store with langmem_impl.

        Reuses LangMem's built-in `prompt` function (which pre-injects
        store.search results into the system prompt) so retrieval semantics
        match the official path. The only difference vs. the preload agent
        is that this one has no `create_manage_memory_tool` — the LLM cannot
        write the MCQ question into memory.
        """
        import os

        from langgraph.prebuilt import create_react_agent
        from langmem import create_search_memory_tool

        from ..utils import load_official_langmem_module

        vendor = load_official_langmem_module()
        return create_react_agent(
            f"openai:{os.getenv('MODEL')}",
            prompt=vendor.prompt,
            tools=[create_search_memory_tool(namespace=("memories",))],
            store=self.langmem_impl.store,
            # No checkpointer: each MCQ is a one-shot query.
        )

    def _snapshot_store(self, query: str = "", limit: int = 200) -> List[Any]:
        try:
            items = self.langmem_impl.store.search(("memories",), query=query or None)
        except TypeError:
            items = self.langmem_impl.store.search(("memories",), query=query)
        return [_jsonable(item) for item in list(items)[:limit]]

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        messages = []
        for msg in context_messages:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()
            if not content:
                continue
            messages.append({"role": role or "user", "content": content})
        self.preload_log["input_messages"] = messages
        if not messages:
            return

        writes = []
        total_batches = (len(messages) + self.preload_batch_size - 1) // self.preload_batch_size
        for batch_index, start in enumerate(range(0, len(messages), self.preload_batch_size), start=1):
            batch = messages[start : start + self.preload_batch_size]
            batch_messages = []
            for message_index, message in enumerate(batch, start=start + 1):
                memory_text = f"Turn {message_index:03d} | {message['role']}: {message['content']}"
                batch_messages.append(memory_text)
            batch_results = []
            for memory_text in batch_messages:
                result = self.langmem_impl.add_memory(memory_text, self.config)
                batch_results.append(_jsonable(result))
            writes.append(
                {
                    "batch_index": batch_index,
                    "messages": batch_messages,
                    "results": batch_results,
                }
            )
        self.preload_log["manager_writes"] = writes
        self.preload_log["store_snapshot"] = self._snapshot_store()

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        # Two retrieval paths, controlled by env LANGMEM_USE_FULL_AGENT:
        #   "1" → full agent (manage + search tools) — original mem0_official path
        #   else → dual-agent Option 2 (search-only query_agent, no write tool)
        use_full_agent = os.environ.get("LANGMEM_USE_FULL_AGENT", "0") == "1"
        self._query_counter += 1
        query_config = {
            "configurable": {
                "thread_id": f"langmem-query-{self.runtime_user_id}-{self._query_counter}"
            }
        }
        try:
            if use_full_agent:
                retrieved_text, _ = self.langmem_impl.search_memory(question, query_config)
                agent_messages = []
            else:
                agent_result = self.query_agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=query_config,
                )
                agent_messages = agent_result.get("messages", [])
                retrieved_text = (
                    getattr(agent_messages[-1], "content", "") if agent_messages else ""
                )
        except Exception as exc:
            agent_messages = []
            retrieved_text = ""
            print(f"[langmem] retrieval failed: {exc}", flush=True)

        # Also keep a raw store snapshot for debug visibility.
        store_hits = self._snapshot_store(query=question, limit=self.memory_limit)

        answer_prompt = OFFICIAL_STYLE_MCq_ANSWER_PROMPT.format(
            memories=retrieved_text or "No relevant memories were retrieved.",
            question=question,
            options=_render_choice_lines(choices),
        )
        response = request_text(
            self.client,
            self.model,
            [{"role": "system", "content": answer_prompt}],
        )
        return {
            "model_response": response,
            "retrieved_memories": store_hits,
            "debug": {
                "retrieved_memories_text": retrieved_text,
                "agent_message_count": len(agent_messages),
            },
        }

    def debug_payload(self) -> Dict[str, Any]:
        return {
            "preload": self.preload_log,
            "memory_limit": self.memory_limit,
            "preload_batch_size": self.preload_batch_size,
            "langmem_source": "vendored_official_langmem",
        }


def build_adapter(
    *,
    args: Any,
    persona_messages: List[Dict[str, str]],
    **_: Any,
) -> LangMemAdapter:
    del persona_messages
    ensure_openai_env(args.api_key_file)
    module = load_official_langmem_module()
    client = load_openai_client(args.api_key_file)

    import os

    os.environ["MODEL"] = args.model
    os.environ["EMBEDDING_MODEL"] = args.embedding_model

    langmem_impl = module.LangMem()
    user_id = Path(args.rendered).stem
    return LangMemAdapter(
        langmem_impl=langmem_impl,
        client=client,
        model=args.model,
        user_id=user_id,
        memory_limit=args.memory_limit,
        preload_batch_size=args.preload_batch_size,
    )
