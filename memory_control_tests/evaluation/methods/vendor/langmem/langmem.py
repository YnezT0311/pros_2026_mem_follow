import os
import time

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool, create_search_memory_tool
from openai import OpenAI

load_dotenv()

client = OpenAI()


def prompt(state):
    """Prepare the messages for the LLM."""
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


class LangMem:
    def __init__(
        self,
    ):
        self.model_id = f"openai:{os.getenv('MODEL')}"
        self.embedding_model_id = f"openai:{os.getenv('EMBEDDING_MODEL')}"
        self.store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": self.embedding_model_id,
            }
        )
        self.checkpointer = MemorySaver()  # Checkpoint graph state

        self.agent = create_react_agent(
            self.model_id,
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
            checkpointer=self.checkpointer,
        )

    def add_memory(self, message, config):
        return self.agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    def search_memory(self, query, config):
        try:
            t1 = time.time()
            response = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
            t2 = time.time()
            return response["messages"][-1].content, t2 - t1
        except Exception as e:
            print(f"Error in search_memory: {e}")
            return "", t2 - t1
