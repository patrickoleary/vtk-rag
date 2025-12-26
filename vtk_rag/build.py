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

Code Map:
    main()                           # CLI entry point
        ├── run_clean()              # clean processed data and indexes (public)
        ├── _check_prerequisites()   # verify raw data, Qdrant, dependencies
        ├── _run_chunking()          # stage 1: chunk raw data
        └── _run_indexing()          # stage 2: build Qdrant indexes
    _print_header/step/warning/error/success()  # terminal output helpers
"""

import argparse
import socket
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient

from .chunking import Chunker
from .config import get_config
from .indexing import Indexer
from .mcp import get_vtk_client
from .rag import RAGClient


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def _print_header(text: str) -> None:
    """Print a section header."""
    print()
    print(Colors.BOLD + Colors.BLUE + "=" * 60 + Colors.END)
    print(Colors.BOLD + Colors.BLUE + text + Colors.END)
    print(Colors.BOLD + Colors.BLUE + "=" * 60 + Colors.END)


def _print_step(text: str) -> None:
    """Print a step indicator."""
    print(Colors.GREEN + f"→ {text}" + Colors.END)


def _print_warning(text: str) -> None:
    """Print a warning message."""
    print(Colors.YELLOW + f"⚠ {text}" + Colors.END)


def _print_error(text: str) -> None:
    """Print an error message."""
    print(Colors.RED + f"✗ {text}" + Colors.END)


def _print_success(text: str) -> None:
    """Print a success message."""
    print(Colors.GREEN + f"✓ {text}" + Colors.END)


def _check_prerequisites(skip_qdrant: bool = False) -> bool:
    """Check if prerequisites are met.

    Args:
        skip_qdrant: If True, don't check for Qdrant (for chunk-only mode).

    Returns:
        True if all prerequisites are met.
    """
    _print_header("Checking Prerequisites")

    issues = []
    base_path = Path(__file__).parent.parent

    # Check raw data files
    _print_step("Checking raw data files...")
    rag = get_config().rag_client
    raw_dir = rag.raw_dir
    raw_files = [
        f'{raw_dir}/{rag.docs_file}',
        f'{raw_dir}/{rag.examples_file}',
        f'{raw_dir}/{rag.tests_file}'
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
        _print_step("Checking Qdrant...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            config = get_config()
            # Parse host/port from URL
            qdrant_url = config.rag_client.qdrant_url
            if "://" in qdrant_url:
                host_port = qdrant_url.split("://")[1]
            else:
                host_port = qdrant_url
            host = host_port.split(":")[0]
            port = int(host_port.split(":")[1]) if ":" in host_port else 6333

            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"  ✓ Qdrant running on {qdrant_url}")
            else:
                issues.append("Qdrant not running")
                _print_warning(f"Qdrant not running on {qdrant_url}")
                print("    Start with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        except Exception as e:
            issues.append(f"Could not check Qdrant: {e}")

    # Check dependencies
    _print_step("Checking dependencies...")
    try:
        from fastembed import SparseTextEmbedding  # noqa: F401
        from qdrant_client import QdrantClient  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        print("  ✓ All dependencies installed")
    except ImportError as e:
        issues.append(f"Missing dependency: {e.name}")
        _print_error(f"Missing dependency: {e.name}")
        print("    Install with: pip install -e .")

    if issues:
        print()
        _print_error("Prerequisites not met:")
        for issue in issues:
            print(f"  • {issue}")
        return False

    print()
    _print_success("All prerequisites met")
    return True


def _run_chunking() -> dict[str, int]:
    """Run the chunking pipeline.

    Returns:
        Dict with chunk counts.
    """
    _print_header("Stage 1: Chunking")

    config = get_config()
    rag_client = RAGClient(config)
    mcp_client = get_vtk_client()
    chunker = Chunker(rag_client, mcp_client)
    return chunker.chunk()


def _run_indexing() -> dict[str, int]:
    """Run the indexing pipeline.

    Returns:
        Dict with index counts.
    """
    _print_header("Stage 2: Indexing")

    config = get_config()
    rag_client = RAGClient(config)
    indexer = Indexer(rag_client)
    return indexer.index()


def run_clean() -> None:
    """Clean processed data and Qdrant collections."""
    _print_header("Cleaning")

    base_path = Path(__file__).parent.parent

    # Clean processed files
    _print_step("Removing processed chunk files...")
    rag = get_config().rag_client
    processed_dir = base_path / rag.chunk_dir
    if processed_dir.exists():
        for file in processed_dir.glob("*.jsonl"):
            file.unlink()
            print(f"  Deleted: {file.name}")

    # Clean Qdrant collections
    _print_step("Removing Qdrant collections...")
    try:
        config = get_config()
        client = QdrantClient(url=config.rag_client.qdrant_url)

        for collection in [config.rag_client.code_collection, config.rag_client.docs_collection]:
            try:
                client.delete_collection(collection)
                print(f"  Deleted: {collection}")
            except Exception:
                print(f"  Skipped: {collection} (not found)")
    except Exception as e:
        _print_warning(f"Could not connect to Qdrant: {e}")

    print()
    _print_success("Clean complete")


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
    _print_header("VTK RAG Build Pipeline")
    print("This will:")
    if run_chunk:
        print("  1. Chunk raw data into semantic chunks")
    if run_index:
        print("  2. Build Qdrant hybrid search indexes")
    print()

    # Check prerequisites
    skip_qdrant = run_chunk and not run_index
    if not _check_prerequisites(skip_qdrant=skip_qdrant):
        if not args.force:
            print()
            print("Use --force to continue anyway")
            sys.exit(1)

    try:
        results = {}

        # Stage 1: Chunking
        if run_chunk:
            chunk_results = _run_chunking()
            results['chunks'] = chunk_results

        # Stage 2: Indexing
        if run_index:
            index_results = _run_indexing()
            results['indexes'] = index_results

        # Summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        _print_header("Build Complete")
        print(f"Total time: {minutes}m {seconds}s")
        print()

        if 'chunks' in results:
            print("Chunks created:")
            chunk_results = results['chunks']
            if 'total_chunks' in chunk_results:
                print(f"  • Total: {chunk_results['total_chunks']:,}")
            if 'code' in chunk_results:
                print(f"  • Code: {chunk_results['code'].get('chunks', 0):,}")
            if 'docs' in chunk_results:
                print(f"  • Docs: {chunk_results['docs'].get('chunks', 0):,}")

        if 'indexes' in results:
            print("Indexes built:")
            for name, count in results['indexes'].items():
                print(f"  • {name}: {count:,}")

        print()
        print("Next steps:")
        print("  • Search: from vtk_rag.retrieval import Retriever")
        print(f"  • Qdrant UI: {get_config().rag_client.qdrant_url}/dashboard")
        print()

    except KeyboardInterrupt:
        print()
        _print_warning("Build cancelled")
        sys.exit(1)
    except Exception as e:
        print()
        _print_error(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
