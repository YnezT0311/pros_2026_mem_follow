"""Vendored MemTree package.

Stripped down for MemoryCtrl: only the in-process tree pieces (structure /
utils / prompt / token_tracker) are needed. main.py / main_mp.py (the
LongMemEval batch runner) are not vendored — the adapter constructs
``GlobalConfig``-shaped state directly and calls ``MemTree`` /
``get_embedding`` / ``search`` on it.

Submodules are eagerly imported here so the adapter can do
``memtree_module.utils.get_embedding = ...`` style monkey-patching. Order
matters: ``config`` defines ``globalconfig`` which the others read at import
time; we leave it as ``None`` and rebind from the adapter before any tree
operation runs.
"""

from . import config  # noqa: F401  — must be first; provides globalconfig=None
from . import prompt  # noqa: F401
from . import token_tracker  # noqa: F401
from . import utils  # noqa: F401
from . import structure  # noqa: F401
