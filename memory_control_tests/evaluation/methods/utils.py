import importlib.util
import sys
import typing
from functools import lru_cache
from pathlib import Path
from types import ModuleType

try:
    from typing_extensions import NotRequired as TypingExtensionsNotRequired
except ImportError:
    TypingExtensionsNotRequired = None


METHODS_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = METHODS_ROOT / "vendor"


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    if not hasattr(typing, "NotRequired") and TypingExtensionsNotRequired is not None:
        typing.NotRequired = TypingExtensionsNotRequired

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def load_official_langmem_module() -> ModuleType:
    return _load_module_from_path(
        "memoryctrl_vendor_langmem_eval",
        VENDOR_ROOT / "langmem" / "langmem.py",
    )


@lru_cache(maxsize=None)
def load_official_amem_module() -> ModuleType:
    return _load_module_from_path(
        "memoryctrl_vendor_amem_eval",
        VENDOR_ROOT / "amem" / "amem.py",
    )


def _load_package_from_path(package_name: str, package_dir: Path) -> ModuleType:
    """Load a multi-file Python package under a unique sys.modules name so its
    relative imports resolve cleanly without polluting the global namespace."""
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        raise ImportError(f"Vendored package missing __init__.py: {init_path}")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load package {package_name} from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def load_official_memoryos_module() -> ModuleType:
    return _load_package_from_path(
        "memoryctrl_vendor_memoryos",
        VENDOR_ROOT / "memoryos",
    )


@lru_cache(maxsize=None)
def load_official_memtree_module() -> ModuleType:
    return _load_package_from_path(
        "memoryctrl_vendor_memtree",
        VENDOR_ROOT / "memtree",
    )
