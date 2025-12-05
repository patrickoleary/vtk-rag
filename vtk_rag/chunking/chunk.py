#!/usr/bin/env python3
"""
Build the VTK RAG corpus by chunking code examples and class/method documentation.

Usage:
    python -m vtk_rag.chunking.chunk

Output:
    data/processed/code-chunks.jsonl  - Code chunks from examples/tests
    data/processed/doc-chunks.jsonl   - Class/method documentation chunks
"""

from .chunker import Chunker


def main() -> None:
    """CLI entry point."""
    chunker = Chunker()
    chunker.chunk_all()


if __name__ == "__main__":
    main()
