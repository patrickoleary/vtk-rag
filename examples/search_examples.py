#!/usr/bin/env python3
"""VTK RAG search examples.

Demonstrates the search capabilities of the Retriever class for finding
VTK code examples and documentation.

Requires a running Qdrant instance with indexed VTK chunks.
Run `uv run vtk-rag build` first to build the index.

Usage:
    uv run python examples/search_examples.py
"""

from vtk_rag.config import get_config
from vtk_rag.rag import RAGClient
from vtk_rag.retrieval import Retriever


def print_results(results, show_content: bool = False):
    """Print search results."""
    for r in results:
        print(f"  [{r.score:.3f}] {r.class_name or r.chunk_id}")
        if r.synopsis:
            synopsis = r.synopsis[:70] + "..." if len(r.synopsis) > 70 else r.synopsis
            print(f"           {synopsis}")
        if show_content and r.content:
            print(f"           Content: {r.content[:100]}...")
        print()


def main():
    """Demonstrate Retriever search capabilities."""
    # Initialize
    config = get_config()
    rag_client = RAGClient(config)
    retriever = Retriever(rag_client)

    print("=" * 70)
    print("VTK RAG Search Examples")
    print("=" * 70)

    # =========================================================================
    # 1. BASIC SEARCH MODES
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. BASIC SEARCH MODES")
    print("=" * 70)

    # Semantic search (dense vectors) - best for natural language
    print("\n1a. Semantic Search (natural language queries)")
    print("-" * 50)
    results = retriever.search(
        "how to create a sphere visualization",
        collection="vtk_code",
        limit=3,
    )
    print_results(results)

    # BM25 search (sparse vectors) - best for exact class/method names
    print("\n1b. BM25 Search (exact class names)")
    print("-" * 50)
    results = retriever.bm25_search(
        "vtkSphereSource",
        collection="vtk_code",
        limit=3,
    )
    print_results(results)

    # Hybrid search (dense + sparse with RRF fusion) - best of both
    print("\n1c. Hybrid Search (combined)")
    print("-" * 50)
    results = retriever.hybrid_search(
        "vtkSphereSource create sphere",
        collection="vtk_code",
        limit=3,
    )
    print_results(results)

    # Multi-vector search (pre-generated queries)
    print("\n1d. Multi-vector Search (queries vector)")
    print("-" * 50)
    results = retriever.search(
        "sphere",
        vector_name="queries",
        limit=3,
    )
    print_results(results)

    # =========================================================================
    # 2. COLLECTION-SPECIFIC SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. COLLECTION-SPECIFIC SEARCH")
    print("=" * 70)

    # Search code examples
    print("\n2a. Search Code Examples")
    print("-" * 50)
    results = retriever.search_code("render volume data", limit=3)
    print_results(results)

    # Search documentation
    print("\n2b. Search Documentation")
    print("-" * 50)
    results = retriever.search_docs("vtkPolyDataMapper", limit=3)
    print_results(results)

    # =========================================================================
    # 3. FILTERED SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. FILTERED SEARCH")
    print("=" * 70)

    # Filter by role
    print("\n3a. Filter by Role (input sources)")
    print("-" * 50)
    results = retriever.search_by_role(
        "geometry",
        role="input",
        limit=3,
    )
    print_results(results)

    # Filter by VTK class
    print("\n3b. Filter by VTK Class")
    print("-" * 50)
    results = retriever.search_by_class("vtkActor", limit=3)
    print_results(results)

    # Filter by data type
    print("\n3c. Filter by Data Type (vtkPolyData input)")
    print("-" * 50)
    results = retriever.search_by_datatype(
        "filter",
        input_type="vtkPolyData",
        limit=3,
    )
    print_results(results)

    # Filter by chunk type (docs only)
    print("\n3d. Filter by Chunk Type (class overviews)")
    print("-" * 50)
    results = retriever.search_by_chunk_type(
        "mapper",
        chunk_type="class_overview",
        limit=3,
    )
    print_results(results)

    # Filter by module
    print("\n3e. Filter by VTK Module")
    print("-" * 50)
    results = retriever.search_by_module(
        "source",
        module="vtkmodules.vtkFiltersSources",
        limit=3,
    )
    print_results(results)

    # =========================================================================
    # 4. ADVANCED FILTERING WITH DICT SYNTAX
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. ADVANCED FILTERING")
    print("=" * 70)

    # Exact match
    print("\n4a. Exact Match Filter")
    print("-" * 50)
    results = retriever.search(
        "render",
        collection="vtk_code",
        filters={"role": "renderer"},
        limit=3,
    )
    print_results(results)

    # Match any (list of values)
    print("\n4b. Match Any Filter (multiple roles)")
    print("-" * 50)
    results = retriever.search(
        "visualization",
        collection="vtk_code",
        filters={"role": ["input", "filter"]},
        limit=3,
    )
    print_results(results)

    # Range filter
    print("\n4c. Range Filter (high visibility)")
    print("-" * 50)
    results = retriever.search(
        "sphere",
        collection="vtk_code",
        filters={"visibility_score": {"gte": 0.8}},
        limit=3,
    )
    print_results(results)

    # Combined filters
    print("\n4d. Combined Filters")
    print("-" * 50)
    results = retriever.search(
        "data processing",
        collection="vtk_code",
        filters={
            "role": "filter",
            "visibility_score": {"gte": 0.5},
        },
        limit=3,
    )
    print_results(results)

    # =========================================================================
    # 5. WORKING WITH RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. WORKING WITH RESULTS")
    print("=" * 70)

    results = retriever.search_code("create cylinder", limit=1)
    if results:
        r = results[0]
        print(f"\nSearchResult fields for: {r.chunk_id}")
        print("-" * 50)
        print(f"  score:            {r.score:.3f}")
        print(f"  collection:       {r.collection}")
        print(f"  class_name:       {r.class_name}")
        print(f"  chunk_type:       {r.chunk_type}")
        print(f"  role:             {r.role}")
        print(f"  synopsis:         {r.synopsis[:50]}..." if r.synopsis else "")
        print(f"  visibility_score: {r.visibility_score}")
        print(f"  input_datatype:   {r.input_datatype}")
        print(f"  output_datatype:  {r.output_datatype}")
        print(f"  content length:   {len(r.content)} chars")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
