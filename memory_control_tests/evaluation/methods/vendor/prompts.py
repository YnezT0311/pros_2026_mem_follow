"""Stub for the `prompts` module that vendor/langmem/langmem.py imports
unqualified at the top of the file (`from prompts import ANSWER_PROMPT`).

The adapter does not actually use ANSWER_PROMPT (we render our own MCQ
answer prompt in `methods/langmem/__init__.py`). This stub keeps the
unqualified import resolvable so that `_load_module_from_path` can load
the vendored file without falling through to an unrelated `prompts.py`
elsewhere on sys.path.
"""

ANSWER_PROMPT = ""
