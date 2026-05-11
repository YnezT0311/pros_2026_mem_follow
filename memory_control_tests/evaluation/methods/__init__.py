from .amem import build_adapter as build_amem_adapter
from .langmem import build_adapter as build_langmem_adapter
from .mem0 import build_adapter as build_mem0_adapter
from .memoryos import build_adapter as build_memoryos_adapter
from .memtree import build_adapter as build_memtree_adapter
from .plain import build_adapter as build_plain_adapter
from .zep import build_adapter as build_zep_adapter


METHOD_BUILDERS = {
    "plain": build_plain_adapter,
    "mem0": build_mem0_adapter,
    "langmem": build_langmem_adapter,
    "amem": build_amem_adapter,
    "zep": build_zep_adapter,
    "memoryos": build_memoryos_adapter,
    "memtree": build_memtree_adapter,
}


def build_method_adapter(method: str, **kwargs):
    if method not in METHOD_BUILDERS:
        raise ValueError(f"Unsupported method: {method}")
    return METHOD_BUILDERS[method](**kwargs)
