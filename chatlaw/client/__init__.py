"""Client-side pipeline entry points for ChatLaw.

Imports are intentionally lazy so running a submodule such as
``python -m chatlaw.client.rag`` does not eagerly import every optional client
dependency during package initialization.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "PromptBuildResult",
    "RagResult",
    "RagResources",
    "TokenizeResult",
    "build_prompt_pipeline",
    "load_chatlaw_tokenizer",
    "load_rag_resources",
    "run_rag",
    "tokenize_prompt",
]


def __getattr__(name: str) -> Any:
    if name in {"PromptBuildResult", "build_prompt_pipeline"}:
        module = import_module("chatlaw.client.build")
    elif name in {"RagResult", "RagResources", "load_rag_resources", "run_rag"}:
        module = import_module("chatlaw.client.rag")
    elif name in {"TokenizeResult", "load_chatlaw_tokenizer", "tokenize_prompt"}:
        module = import_module("chatlaw.client.tokenizer")
    else:
        raise AttributeError(f"module 'chatlaw.client' has no attribute {name!r}")

    return getattr(module, name)
