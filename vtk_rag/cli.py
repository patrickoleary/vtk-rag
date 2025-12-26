#!/usr/bin/env python3
"""
VTK RAG Command Line Interface

Unified CLI for chunking, indexing, and retrieval operations.

Usage:
    python -m vtk_rag chunk              # Process raw data into chunks
    python -m vtk_rag index              # Build Qdrant indexes
    python -m vtk_rag build              # Full pipeline (chunk + index)
    python -m vtk_rag clean              # Remove processed data and indexes
    python -m vtk_rag search "query"     # Search code and docs

Code Map:
    main()                           # CLI entry point
        ├── cmd_chunk()              # chunk subcommand
        ├── cmd_index()              # index subcommand
        ├── cmd_build()              # build subcommand (calls build.main)
        ├── cmd_clean()              # clean subcommand (calls build.run_clean)
        └── cmd_search()             # search subcommand
"""

import argparse
import sys

from .build import main as build_main
from .build import run_clean
from .chunking import Chunker
from .config import get_config
from .indexing import Indexer
from .mcp import get_vtk_client
from .rag import RAGClient
from .retrieval import Retriever


def cmd_chunk(args: argparse.Namespace) -> None:
    """Run the chunking pipeline."""
    config = get_config()
    rag_client = RAGClient(config)
    mcp_client = get_vtk_client()
    chunker = Chunker(rag_client, mcp_client)
    chunker.chunk()


def cmd_index(args: argparse.Namespace) -> None:
    """Run the indexing pipeline."""
    config = get_config()
    rag_client = RAGClient(config)
    indexer = Indexer(rag_client)
    indexer.index()


def cmd_build(args: argparse.Namespace) -> None:
    """Run the full build pipeline."""
    # Pass through args
    sys.argv = ['build']
    if args.force:
        sys.argv.append('--force')
    build_main()


def cmd_clean(args: argparse.Namespace) -> None:
    """Clean processed data and indexes."""
    run_clean()


def cmd_search(args: argparse.Namespace) -> None:
    """Search code and documentation."""
    config = get_config()
    rag_client = RAGClient(config)
    retriever = Retriever(rag_client)

    # Determine collection
    if args.code:
        collection = "vtk_code"
    elif args.docs:
        collection = "vtk_docs"
    else:
        collection = None  # Search both

    # Determine search mode
    if args.hybrid:
        search_fn = retriever.hybrid_search
    elif args.bm25:
        search_fn = retriever.bm25_search
    else:
        search_fn = retriever.search

    # Build filters
    filters = {}
    if args.role:
        filters["role"] = args.role
    if args.type:
        filters["type"] = args.type
    if args.class_name:
        filters["class_name"] = args.class_name

    # Execute search
    if collection:
        results = search_fn(args.query, collection=collection, limit=args.limit,
                           filters=filters if filters else None)
    else:
        # Search both collections
        code_results = search_fn(args.query, collection="vtk_code", limit=args.limit,
                                 filters=filters if filters else None)
        doc_results = search_fn(args.query, collection="vtk_docs", limit=args.limit,
                                filters=filters if filters else None)
        results = sorted(code_results + doc_results, key=lambda r: r.score, reverse=True)[:args.limit]

    # Display results
    print(f"\nSearch: \"{args.query}\"")
    print(f"Results: {len(results)}")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.class_name or 'N/A'} ({r.collection})")
        print(f"    Score: {r.score:.4f}")
        print(f"    Type: {r.chunk_type}")
        if r.synopsis:
            synopsis = r.synopsis[:100] + "..." if len(r.synopsis) > 100 else r.synopsis
            print(f"    Synopsis: {synopsis}")
        if r.role:
            print(f"    Role: {r.role}")
        if args.verbose:
            content = r.content[:200] + "..." if len(r.content) > 200 else r.content
            print(f"    Content: {content}")

    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vtk-rag",
        description="VTK RAG - Chunking, Indexing, and Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Process raw data into chunks")
    chunk_parser.set_defaults(func=cmd_chunk)

    # index command
    index_parser = subparsers.add_parser("index", help="Build Qdrant indexes")
    index_parser.set_defaults(func=cmd_index)

    # build command
    build_parser = subparsers.add_parser("build", help="Full pipeline (chunk + index)")
    build_parser.add_argument("--force", action="store_true", help="Continue if prerequisites fail")
    build_parser.set_defaults(func=cmd_build)

    # clean command
    clean_parser = subparsers.add_parser("clean", help="Remove processed data and indexes")
    clean_parser.set_defaults(func=cmd_clean)

    # search command
    search_parser = subparsers.add_parser("search", help="Search code and documentation")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of results (default: 5)")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show content preview")

    # Search mode
    mode_group = search_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--hybrid", action="store_true", help="Use hybrid search (dense + BM25)")
    mode_group.add_argument("--bm25", action="store_true", help="Use BM25 keyword search")

    # Collection
    coll_group = search_parser.add_mutually_exclusive_group()
    coll_group.add_argument("--code", action="store_true", help="Search code chunks only")
    coll_group.add_argument("--docs", action="store_true", help="Search doc chunks only")

    # Filters
    search_parser.add_argument("--role", help="Filter by role (e.g., source_geometric)")
    search_parser.add_argument("--type", help="Filter by chunk type")
    search_parser.add_argument("--class-name", dest="class_name", help="Filter by VTK class name")

    search_parser.set_defaults(func=cmd_search)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
