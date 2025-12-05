#!/usr/bin/env python3
"""
Index VTK RAG chunks into Qdrant for hybrid search.

Usage:
    python -m vtk_rag.indexing.index

Output:
    vtk_code collection - Code chunks from examples/tests
    vtk_docs collection - Class/method documentation chunks
"""

from .indexer import Indexer


def main() -> None:
    """CLI entry point."""
    indexer = Indexer()
    indexer.index_all()


if __name__ == "__main__":
    main()
