#!/usr/bin/env python3
"""Example usage of the VTK RAG Retriever.

This script demonstrates the various search capabilities of the Retriever class.
Requires a running Qdrant instance with indexed VTK chunks.

Usage:
    python examples/retriever_usage.py
"""

from vtk_rag.config import load_config
from vtk_rag.rag import RAGClient
from vtk_rag.retrieval import Retriever


def main():
    """Demonstrate Retriever search capabilities."""
    config = load_config()
    rag_client = RAGClient(config.rag_client)
    retriever = Retriever(rag_client)

    print("=" * 60)
    print("VTK RAG Retriever Examples")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Semantic Search (dense vectors)
    # -------------------------------------------------------------------------
    print("\n1. Semantic Search (dense vectors)")
    print("-" * 40)
    results = retriever.search("create a sphere", collection="vtk_code", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name}: {r.synopsis[:60]}...")

    # -------------------------------------------------------------------------
    # BM25 Search (sparse vectors)
    # -------------------------------------------------------------------------
    print("\n2. BM25 Search (sparse vectors)")
    print("-" * 40)
    results = retriever.bm25_search("vtkSphereSource", collection="vtk_code", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name}: {r.synopsis[:60]}...")

    # -------------------------------------------------------------------------
    # Hybrid Search (dense + sparse with RRF fusion)
    # -------------------------------------------------------------------------
    print("\n3. Hybrid Search (RRF fusion)")
    print("-" * 40)
    results = retriever.hybrid_search("vtkSphereSource", collection="vtk_docs", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name}: {r.synopsis[:60]}...")

    # -------------------------------------------------------------------------
    # Multi-vector Search (pre-generated queries)
    # -------------------------------------------------------------------------
    print("\n4. Multi-vector Search (queries vector)")
    print("-" * 40)
    results = retriever.search("sphere", vector_name="queries", limit=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name}: {r.synopsis[:60]}...")

    # -------------------------------------------------------------------------
    # Filtered Search
    # -------------------------------------------------------------------------
    print("\n5. Filtered Search (by role)")
    print("-" * 40)
    results = retriever.search(
        "render pipeline",
        collection="vtk_code",
        filters={"role": "input"},
        limit=3,
    )
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name} ({r.role}): {r.synopsis[:50]}...")

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    print("\n6. Convenience Methods")
    print("-" * 40)

    # Search code collection
    print("\n  search_code('render pipeline'):")
    results = retriever.search_code("render pipeline", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name}")

    # Search docs collection
    print("\n  search_docs('vtkPolyDataMapper'):")
    results = retriever.search_docs("vtkPolyDataMapper", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name}")

    # Search by class name
    print("\n  search_by_class('vtkSphereSource'):")
    results = retriever.search_by_class("vtkSphereSource", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name} ({r.collection})")

    # Search by role
    print("\n  search_by_role('geometry', role='input'):")
    results = retriever.search_by_role("geometry", role="input", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name} ({r.role})")

    # Search by datatype
    print("\n  search_by_datatype('filter', input_type='vtkPolyData'):")
    results = retriever.search_by_datatype("filter", input_type="vtkPolyData", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name}")

    # Search by chunk type (docs only)
    print("\n  search_by_chunk_type('API', chunk_type='class_overview'):")
    results = retriever.search_by_chunk_type("API", chunk_type="class_overview", limit=2)
    for r in results:
        print(f"    [{r.score:.3f}] {r.class_name} ({r.chunk_type})")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
