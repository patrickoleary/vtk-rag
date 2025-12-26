#!/usr/bin/env python3
"""Show one sample chunk from each role with metadata and content.

Usage:
    uv run python examples/show_chunks_by_role.py
    uv run python examples/show_chunks_by_role.py --collection vtk_code
    uv run python examples/show_chunks_by_role.py --collection vtk_docs
"""

import argparse
from collections import defaultdict

from vtk_rag.config import get_config
from vtk_rag.rag import RAGClient


def get_chunks_by_role(rag_client: RAGClient, collection: str, limit_per_role: int = 1):
    """Fetch chunks grouped by role from Qdrant."""
    client = rag_client.qdrant_client

    # Scroll through all points to find unique roles
    roles_seen: dict[str, list] = defaultdict(list)
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            break

        for point in results:
            payload = point.payload or {}
            role = payload.get("role", "unknown")

            if len(roles_seen[role]) < limit_per_role:
                roles_seen[role].append({
                    "id": point.id,
                    "payload": payload,
                })

        if offset is None:
            break

    return dict(roles_seen)


def print_chunk(role: str, chunk: dict, collection: str):
    """Print a chunk with its metadata and content."""
    payload = chunk["payload"]

    print("=" * 80)
    print(f"ROLE: {role}")
    print(f"Collection: {collection}")
    print("=" * 80)

    # Metadata section
    print("\n--- METADATA ---")
    metadata_fields = [
        "chunk_id", "chunk_type", "vtk_class_names", "class_name",
        "variable_name", "function_name", "example_id",
        "visibility_score", "input_datatype", "output_datatype",
    ]
    for field in metadata_fields:
        if field in payload and payload[field]:
            value = payload[field]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            print(f"  {field}: {value}")

    # Synopsis/action phrase
    print("\n--- SYNOPSIS ---")
    if payload.get("synopsis"):
        print(f"  {payload['synopsis']}")
    if payload.get("action_phrase"):
        print(f"  Action: {payload['action_phrase']}")

    # Queries
    if payload.get("queries"):
        print("\n--- QUERIES ---")
        queries = payload["queries"]
        if isinstance(queries, list):
            for q in queries[:3]:  # Show first 3
                print(f"  • {q}")
            if len(queries) > 3:
                print(f"  ... and {len(queries) - 3} more")

    # Content
    print("\n--- CONTENT ---")
    content = payload.get("content", "")
    # Truncate if too long
    if len(content) > 1500:
        print(content[:1500])
        print(f"\n... [truncated, {len(content)} chars total]")
    else:
        print(content)

    print()


def main():
    parser = argparse.ArgumentParser(description="Show sample chunks from each role")
    parser.add_argument("--collection", choices=["vtk_code", "vtk_docs", "both"],
                        default="both", help="Collection to query")
    parser.add_argument("--limit", type=int, default=1,
                        help="Number of chunks per role (default: 1)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    config = get_config()
    rag_client = RAGClient(config)

    collections = []
    if args.collection in ("vtk_code", "both"):
        collections.append("vtk_code")
    if args.collection in ("vtk_docs", "both"):
        collections.append("vtk_docs")

    # Redirect output to file if specified
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file

    for collection in collections:
        print(f"\n{'#' * 80}")
        print(f"# COLLECTION: {collection}")
        print(f"{'#' * 80}\n")

        chunks_by_role = get_chunks_by_role(rag_client, collection, args.limit)

        print(f"Found {len(chunks_by_role)} unique roles:\n")
        for role in sorted(chunks_by_role.keys()):
            print(f"  • {role} ({len(chunks_by_role[role])} sample(s))")
        print()

        for role in sorted(chunks_by_role.keys()):
            for chunk in chunks_by_role[role]:
                print_chunk(role, chunk, collection)

    # Restore stdout and close file
    if output_file:
        sys.stdout = original_stdout
        output_file.close()
        print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
