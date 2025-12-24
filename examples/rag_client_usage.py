#!/usr/bin/env python3
"""RAGClient usage examples.

Demonstrates how to use RAGClient for VTK code example retrieval.
"""

from vtk_rag.config import load_config
from vtk_rag.rag import RAGClient
from vtk_rag.retrieval import Retriever


def basic_search():
    """Basic code search."""
    config = load_config()
    rag_client = RAGClient(config.rag_client)
    retriever = Retriever(rag_client)
    
    results = retriever.search(
        query="create sphere visualization",
        collection=rag_client.code_collection,
    )
    for r in results:
        print(f"{r.class_name}: {r.synopsis}")


def search_with_options():
    """Search with custom options."""
    config = load_config()
    rag_client = RAGClient(config.rag_client)
    retriever = Retriever(rag_client)
    
    # Use hybrid search with custom limit
    results = retriever.hybrid_search(
        query="render volume data",
        collection=rag_client.code_collection,
        limit=10,
    )
    
    for r in results:
        print(f"[{r.score:.3f}] {r.source_file}")
        print(f"  {r.synopsis}")
        print()


def search_docs():
    """Search documentation."""
    config = load_config()
    rag_client = RAGClient(config.rag_client)
    retriever = Retriever(rag_client)
    
    results = retriever.search(
        query="vtkActor properties",
        collection=rag_client.docs_collection,
    )
    for r in results:
        print(f"{r.class_name}: {r.content[:100]}...")


def with_custom_config():
    """Use custom configuration."""
    config = load_config()
    # Modify config if needed
    config.rag_client.top_k = 3
    
    rag_client = RAGClient(config.rag_client)
    retriever = Retriever(rag_client)
    results = retriever.search("filter pipeline", limit=rag_client.top_k)
    
    print(f"Found {len(results)} results")


if __name__ == "__main__":
    print("=== Basic Search ===")
    basic_search()
    
    print("\n=== Search with Options ===")
    search_with_options()
    
    print("\n=== Search Docs ===")
    search_docs()
    
    print("\n=== Custom Config ===")
    with_custom_config()
