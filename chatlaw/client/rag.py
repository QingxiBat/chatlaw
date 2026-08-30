"""Standalone RAG pipeline for ChatLaw client code.

This module wraps the existing low-level retrieval helpers into one complete
operation: resource preparation, vectorstore/index loading, law retrieval, and
optional CLI output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import markdown

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from chatlaw.dataloader import download_resources
from launcher import get_resources_path


RAG_DEPENDENCY_HINT = (
    "RAG requires langchain-huggingface, langchain-community, faiss-cpu, "
    "and sentence-transformers. Install project dependencies with "
    "`python -m pip install -r requirements.txt`."
)


def _common_utils():
    try:
        from chatlaw.client.utils import common_utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{exc}. {RAG_DEPENDENCY_HINT}") from exc
    return common_utils


@dataclass(frozen=True)
class RagResources:
    """Loaded resources needed for one RAG retrieval pass."""

    vectorstore: Any
    exact_index: dict[str, Any]
    law_name_candidates: list[str]


@dataclass(frozen=True)
class RagResult:
    """Result returned by the full RAG pipeline."""

    query: str
    docs: list[Any]


@lru_cache(maxsize=1)
def load_rag_resources() -> RagResources:
    """Download missing vectorstore files and load all RAG resources."""
    common_utils = _common_utils()
    download_resources(resource_type="vectorstore")

    vectorstore_path = os.path.join(get_resources_path(), "vectorstore", "law_faiss")
    vectorstore = common_utils.load_vectorstore(vectorstore_path)
    exact_index = common_utils.load_exact_index()
    law_name_candidates = common_utils.build_law_name_candidates(exact_index)

    return RagResources(
        vectorstore=vectorstore,
        exact_index=exact_index,
        law_name_candidates=law_name_candidates,
    )


def run_rag(
    query: str,
    *,
    exact_faiss_k: int = 1,
    faiss_only_k: int = 5,
    resources: RagResources | None = None,
) -> RagResult:
    """Run the complete RAG retrieval operation for a user query."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    common_utils = _common_utils()
    resources = resources or load_rag_resources()
    docs = common_utils.retrieve_laws(
        vectorstore=resources.vectorstore,
        query=query,
        exact_index=resources.exact_index,
        law_name_candidates=resources.law_name_candidates,
        exact_faiss_k=exact_faiss_k,
        faiss_only_k=faiss_only_k,
    )
    return RagResult(query=query, docs=docs)


def extract_doc_fields(item: Any) -> tuple[str, str, str]:
    """Extract law name, article number, and content from supported doc types."""
    if hasattr(item, "metadata"):
        metadata = item.metadata or {}
        law_name = (
            metadata.get("law_name_raw")
            or metadata.get("law_name")
            or metadata.get("law_name_norm")
            or "未知法律"
        )
        article = metadata.get("article", "")
        content = metadata.get("content", "") or getattr(item, "page_content", "")
        return law_name, article, content

    if isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or "未知法律"
        )
        article = item.get("article", "")
        content = item.get("content", "")
        return law_name, article, content

    return "未知法律", "", str(item)


def docs_to_markdown(docs: list[Any]) -> str:
    """Format retrieved documents as Markdown."""
    if not docs:
        return "未检索到相关法律条文。"

    blocks = []
    for i, doc in enumerate(docs, 1):
        law_name, article, content = extract_doc_fields(doc)
        blocks.append(f"### {i}. 《{law_name}》{article}\n\n{content}")

    return "\n\n---\n\n".join(blocks)


def docs_to_html(docs: list[Any]) -> str:
    """Format retrieved documents as HTML for the Gradio panel."""
    common_utils = _common_utils()
    rendered = markdown.markdown(docs_to_markdown(docs), extensions=["fenced_code", "tables"])
    html = common_utils.render_mathml_from_latex(rendered)
    return f"<div>{html}</div>"


def docs_to_jsonable(docs: list[Any]) -> list[dict[str, str]]:
    """Convert retrieval documents to plain JSON-serializable records."""
    records = []
    for doc in docs:
        law_name, article, content = extract_doc_fields(doc)
        records.append(
            {
                "law_name": law_name,
                "article": article,
                "content": content,
            }
        )
    return records


def run_rag_as_json(query: str, **kwargs: Any) -> dict[str, Any]:
    """Run RAG and return a JSON-serializable payload."""
    result = run_rag(query, **kwargs)
    return {
        "query": result.query,
        "docs": docs_to_jsonable(result.docs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChatLaw RAG as a standalone step.")
    parser.add_argument("--query", required=True, help="User question to retrieve law articles for.")
    parser.add_argument("--exact-faiss-k", type=int, default=1)
    parser.add_argument("--faiss-only-k", type=int, default=5)
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    try:
        result = run_rag(
            args.query,
            exact_faiss_k=args.exact_faiss_k,
            faiss_only_k=args.faiss_only_k,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.format == "json":
        payload = {"query": result.query, "docs": docs_to_jsonable(result.docs)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(docs_to_markdown(result.docs))


if __name__ == "__main__":
    main()
