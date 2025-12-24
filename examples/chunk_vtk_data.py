#!/usr/bin/env python3
"""Example: Chunk VTK data for RAG indexing.

This script demonstrates how to use the Chunker class to process
VTK examples, tests, and documentation into semantic chunks.
"""

from vtk_rag.chunking import Chunker
from vtk_rag.config import get_config
from vtk_rag.mcp import get_vtk_client
from vtk_rag.rag import RAGClient


def main() -> None:
    """Run the chunking pipeline."""
    config = get_config()
    rag_client = RAGClient(config.rag_client)
    mcp_client = get_vtk_client()

    chunker = Chunker(rag_client, mcp_client)
    stats = chunker.chunk()

    print(f"\nGenerated {stats['total_chunks']} total chunks")


if __name__ == "__main__":
    main()
