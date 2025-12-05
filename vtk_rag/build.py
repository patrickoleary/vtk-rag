#!/usr/bin/env python3
"""
VTK RAG Build Pipeline

Runs the complete pipeline from raw data to searchable index:
1. Chunking - Process raw data into semantic chunks
2. Indexing - Build Qdrant collections with hybrid search

Usage:
    python -m vtk_rag.build           # Run full pipeline
    python -m vtk_rag.build --chunk   # Chunking only
    python -m vtk_rag.build --index   # Indexing only
    python -m vtk_rag.build --clean   # Clean processed data and indexes

Prerequisites:
    - Raw data files in data/raw/
    - Qdrant running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""

import argparse
import socket
import sys
import time
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a section header."""
    print()
    print(Colors.BOLD + Colors.BLUE + "=" * 60 + Colors.END)
    print(Colors.BOLD + Colors.BLUE + text + Colors.END)
    print(Colors.BOLD + Colors.BLUE + "=" * 60 + Colors.END)


def print_step(text: str) -> None:
    """Print a step indicator."""
    print(Colors.GREEN + f"→ {text}" + Colors.END)


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(Colors.YELLOW + f"⚠ {text}" + Colors.END)


def print_error(text: str) -> None:
    """Print an error message."""
    print(Colors.RED + f"✗ {text}" + Colors.END)


def print_success(text: str) -> None:
    """Print a success message."""
    print(Colors.GREEN + f"✓ {text}" + Colors.END)


def check_prerequisites(skip_qdrant: bool = False) -> bool:
    """Check if prerequisites are met.

    Args:
        skip_qdrant: If True, don't check for Qdrant (for chunk-only mode).

    Returns:
        True if all prerequisites are met.
    """
    print_header("Checking Prerequisites")

    issues = []
    base_path = Path(__file__).parent.parent

    # Check raw data files
    print_step("Checking raw data files...")
    raw_files = [
        'data/raw/vtk-python-docs.jsonl',
        'data/raw/vtk-python-examples.jsonl',
        'data/raw/vtk-python-tests.jsonl'
    ]

    for file in raw_files:
        path = base_path / file
        if not path.exists():
            issues.append(f"Missing: {file}")
            print(f"  ✗ {file}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.1f} MB)")

    # Check Qdrant (unless skipped)
    if not skip_qdrant:
        print_step("Checking Qdrant...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 6333))
            sock.close()
            if result == 0:
                print("  ✓ Qdrant running on localhost:6333")
            else:
                issues.append("Qdrant not running")
                print_warning("Qdrant not running on localhost:6333")
                print("    Start with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        except Exception as e:
            issues.append(f"Could not check Qdrant: {e}")

    # Check dependencies
    print_step("Checking dependencies...")
    try:
        from fastembed import SparseTextEmbedding  # noqa: F401
        from qdrant_client import QdrantClient  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        print("  ✓ All dependencies installed")
    except ImportError as e:
        issues.append(f"Missing dependency: {e.name}")
        print_error(f"Missing dependency: {e.name}")
        print("    Install with: pip install -e .")

    if issues:
        print()
        print_error("Prerequisites not met:")
        for issue in issues:
            print(f"  • {issue}")
        return False

    print()
    print_success("All prerequisites met")
    return True


def run_chunking() -> dict[str, int]:
    """Run the chunking pipeline.

    Returns:
        Dict with chunk counts.
    """
    print_header("Stage 1: Chunking")

    from .chunking import Chunker

    chunker = Chunker()
    return chunker.chunk_all()


def run_indexing() -> dict[str, int]:
    """Run the indexing pipeline.

    Returns:
        Dict with index counts.
    """
    print_header("Stage 2: Indexing")

    from .indexing import Indexer

    indexer = Indexer()
    return indexer.index_all()


def run_clean() -> None:
    """Clean processed data and Qdrant collections."""
    print_header("Cleaning")

    base_path = Path(__file__).parent.parent

    # Clean processed files
    print_step("Removing processed chunk files...")
    processed_dir = base_path / "data/processed"
    if processed_dir.exists():
        for file in processed_dir.glob("*.jsonl"):
            file.unlink()
            print(f"  Deleted: {file.name}")

    # Clean Qdrant collections
    print_step("Removing Qdrant collections...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")

        for collection in ["vtk_code", "vtk_docs"]:
            try:
                client.delete_collection(collection)
                print(f"  Deleted: {collection}")
            except Exception:
                print(f"  Skipped: {collection} (not found)")
    except Exception as e:
        print_warning(f"Could not connect to Qdrant: {e}")

    print()
    print_success("Clean complete")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VTK RAG Build Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vtk_rag.build           # Full pipeline (chunk + index)
  python -m vtk_rag.build --chunk   # Chunking only
  python -m vtk_rag.build --index   # Indexing only (requires chunks)
  python -m vtk_rag.build --clean   # Remove processed data and indexes
        """
    )
    parser.add_argument('--chunk', action='store_true',
                        help='Run chunking only')
    parser.add_argument('--index', action='store_true',
                        help='Run indexing only')
    parser.add_argument('--clean', action='store_true',
                        help='Clean processed data and indexes')
    parser.add_argument('--force', action='store_true',
                        help='Continue even if prerequisites fail')
    args = parser.parse_args()

    start_time = time.time()

    # Handle clean
    if args.clean:
        run_clean()
        return

    # Determine what to run
    run_chunk = args.chunk or (not args.chunk and not args.index)
    run_index = args.index or (not args.chunk and not args.index)

    # Print plan
    print_header("VTK RAG Build Pipeline")
    print("This will:")
    if run_chunk:
        print("  1. Chunk raw data into semantic chunks")
    if run_index:
        print("  2. Build Qdrant hybrid search indexes")
    print()

    # Check prerequisites
    skip_qdrant = run_chunk and not run_index
    if not check_prerequisites(skip_qdrant=skip_qdrant):
        if not args.force:
            print()
            print("Use --force to continue anyway")
            sys.exit(1)

    try:
        results = {}

        # Stage 1: Chunking
        if run_chunk:
            chunk_results = run_chunking()
            results['chunks'] = chunk_results

        # Stage 2: Indexing
        if run_index:
            index_results = run_indexing()
            results['indexes'] = index_results

        # Summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print_header("Build Complete")
        print(f"Total time: {minutes}m {seconds}s")
        print()

        if 'chunks' in results:
            print("Chunks created:")
            for name, count in results['chunks'].items():
                print(f"  • {name}: {count:,}")

        if 'indexes' in results:
            print("Indexes built:")
            for name, count in results['indexes'].items():
                print(f"  • {name}: {count:,}")

        print()
        print("Next steps:")
        print("  • Search: from vtk_rag.retrieval import Retriever")
        print("  • Qdrant UI: http://localhost:6333/dashboard")
        print()

    except KeyboardInterrupt:
        print()
        print_warning("Build cancelled")
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
