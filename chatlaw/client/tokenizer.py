"""Standalone tokenization pipeline for ChatLaw client code."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from chatlaw.client.build import build_prompt_pipeline
from chatlaw.dataloader import download_resources
from launcher import get_resources_path


TOKENIZER_DEPENDENCY_HINT = (
    "Tokenization requires transformers. Install project dependencies with "
    "`python -m pip install -r requirements.txt`."
)


def _auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{exc}. {TOKENIZER_DEPENDENCY_HINT}") from exc
    return AutoTokenizer


@dataclass(frozen=True)
class TokenizeResult:
    """Result returned by the full tokenization operation."""

    prompt: str
    templated_prompt: str
    model_inputs: Any


@lru_cache(maxsize=1)
def load_chatlaw_tokenizer() -> Any:
    """Download missing tokenizer files and load the local tokenizer."""
    download_resources(resource_type="tokenizer")
    AutoTokenizer = _auto_tokenizer()
    tokenizer_path = os.path.join(get_resources_path(), "tokenizer").replace("\\", "/")
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def tokenize_prompt(
    prompt: str,
    *,
    return_tensors: str = "pt",
    tokenizer: Any | None = None,
) -> TokenizeResult:
    """Apply Qwen chat template and tokenize the final prompt."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    tokenizer = tokenizer or load_chatlaw_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([templated_prompt], return_tensors=return_tensors)

    return TokenizeResult(
        prompt=prompt,
        templated_prompt=templated_prompt,
        model_inputs=model_inputs,
    )


def tokenize_question(
    question: str,
    *,
    return_tensors: str = "pt",
    exact_faiss_k: int = 1,
    faiss_only_k: int = 5,
) -> TokenizeResult:
    """Run RAG, build the final prompt, and tokenize it in one call."""
    prompt_result = build_prompt_pipeline(
        question,
        exact_faiss_k=exact_faiss_k,
        faiss_only_k=faiss_only_k,
    )
    return tokenize_prompt(prompt_result.prompt, return_tensors=return_tensors)


def summarize_model_inputs(model_inputs: Any) -> dict[str, dict[str, Any]]:
    """Return serializable tensor metadata for quick CLI inspection."""
    summary = {}
    for key, value in model_inputs.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        summary[key] = {
            "shape": list(shape) if shape is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
        }
    return summary


def tokenize_prompt_as_json(prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Tokenize a prompt and return a JSON-serializable summary."""
    result = tokenize_prompt(prompt, **kwargs)
    return {
        "prompt_length": len(result.prompt),
        "templated_prompt_length": len(result.templated_prompt),
        "model_inputs": summarize_model_inputs(result.model_inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChatLaw tokenizer as a standalone step.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Final prompt to tokenize.")
    source.add_argument("--question", help="User question; RAG and prompt build will run first.")
    parser.add_argument(
        "--mode",
        choices=["pt", "ms"],
        default="pt",
        help="pt returns PyTorch tensors; ms returns NumPy tensors.",
    )
    parser.add_argument("--exact-faiss-k", type=int, default=1)
    parser.add_argument("--faiss-only-k", type=int, default=5)
    parser.add_argument(
        "--format",
        choices=["summary", "json"],
        default="summary",
        help="Output a readable summary or JSON metadata.",
    )
    args = parser.parse_args()

    return_tensors = "np" if args.mode == "ms" else "pt"
    try:
        if args.prompt:
            result = tokenize_prompt(args.prompt, return_tensors=return_tensors)
        else:
            result = tokenize_question(
                args.question,
                return_tensors=return_tensors,
                exact_faiss_k=args.exact_faiss_k,
                faiss_only_k=args.faiss_only_k,
            )
    except (ModuleNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    payload = {
        "prompt_length": len(result.prompt),
        "templated_prompt_length": len(result.templated_prompt),
        "model_inputs": summarize_model_inputs(result.model_inputs),
    }

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"prompt_length: {payload['prompt_length']}")
        print(f"templated_prompt_length: {payload['templated_prompt_length']}")
        for name, info in payload["model_inputs"].items():
            print(f"{name}: shape={info['shape']} dtype={info['dtype']}")


if __name__ == "__main__":
    main()
