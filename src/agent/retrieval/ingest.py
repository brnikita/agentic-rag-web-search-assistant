"""Ingest the internal docs corpus into the vector store.

Run with `uv run agent-ingest` (add `--reset` to rebuild from scratch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import Settings, get_settings
from agent.retrieval.store import get_vector_store

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def load_documents(docs_dir: Path) -> list[Document]:
    """Read every supported file under `docs_dir` into a Document.

    `source` is stored relative to the docs dir so citations stay stable
    regardless of where the repo is checked out.
    """
    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.relative_to(docs_dir).as_posix(),
                    "title": path.stem.replace("-", " ").replace("_", " "),
                },
            )
        )
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def ingest(settings: Settings | None = None, *, reset: bool = False) -> int:
    """Chunk and embed the docs corpus. Returns the number of chunks written."""
    settings = settings or get_settings()

    documents = load_documents(settings.docs_dir)
    if not documents:
        print(f"No documents found under {settings.docs_dir}", file=sys.stderr)
        return 0

    store = get_vector_store(settings)
    if reset:
        # Drop the collection so re-ingesting edited docs doesn't leave stale
        # chunks behind; Chroma has no upsert-by-source primitive.
        store.reset_collection()

    chunks = split_documents(documents, settings)
    store.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {len(documents)} documents.")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest internal docs into the vector store.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing chunks before ingesting (use after editing docs).",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
