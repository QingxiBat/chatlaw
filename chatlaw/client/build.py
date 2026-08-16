"""Standalone prompt-building pipeline for ChatLaw client code."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from chatlaw.client.rag import RagResult, docs_to_jsonable, run_rag


BUILD_DEPENDENCY_HINT = (
    "Prompt building imports the original common_utils.build_prompt helper. "
    "If this fails while importing RAG dependencies, install project "
    "dependencies with `python -m pip install -r requirements.txt`."
)


def _build_prompt(question: str, docs: list[Any]) -> str:
    try:
        from chatlaw.client.utils.common_utils import build_prompt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{exc}. {BUILD_DEPENDENCY_HINT}") from exc
    return build_prompt(question, docs)


@dataclass(frozen=True)
class PromptBuildResult:
    """Result returned by the full prompt-building operation."""

    question: str
    docs: list[Any]
    prompt: str


def build_prompt_pipeline(
    question: str,
    docs: list[Any] | None = None,
    *,
    rag_result: RagResult | None = None,
    exact_faiss_k: int = 1,
    faiss_only_k: int = 5,
) -> PromptBuildResult:
    """Build the final model prompt, running RAG first when docs are absent."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    if rag_result is not None:
        docs = rag_result.docs
        question = rag_result.query
    elif docs is None:
        rag_result = run_rag(
            question,
            exact_faiss_k=exact_faiss_k,
            faiss_only_k=faiss_only_k,
        )
        docs = rag_result.docs

    final_prompt = _build_prompt(question, docs or [])
    return PromptBuildResult(question=question, docs=docs or [], prompt=final_prompt)


def build_prompt_as_json(question: str, **kwargs: Any) -> dict[str, Any]:
    """Build a prompt and return a JSON-serializable payload."""
    result = build_prompt_pipeline(question, **kwargs)
    return {
        "question": result.question,
        "docs": docs_to_jsonable(result.docs),
        "prompt": result.prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChatLaw final prompt as a standalone step.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--exact-faiss-k", type=int, default=1)
    parser.add_argument("--faiss-only-k", type=int, default=5)
    parser.add_argument(
        "--format",
        choices=["prompt", "json"],
        default="prompt",
        help="Output only the prompt or a JSON payload with docs.",
    )
    args = parser.parse_args()

    try:
        result = build_prompt_pipeline(
            args.question,
            exact_faiss_k=args.exact_faiss_k,
            faiss_only_k=args.faiss_only_k,
        )
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.format == "json":
        payload = {
            "question": result.question,
            "docs": docs_to_jsonable(result.docs),
            "prompt": result.prompt,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(result.prompt)


if __name__ == "__main__":
    main()
