"""Chunker class for building VTK RAG corpus.

Code Map:
    Chunker
        chunk()                      # public API - chunk all sources
            ├── chunk_code()         # code examples and tests
            │       └── _process_code_file()
            └── chunk_docs()         # API documentation
                    └── _process_doc_file()
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vtk_rag.mcp import VTKClient
from vtk_rag.rag import RAGClient

from .code import CodeChunker
from .doc import DocChunker


class Chunker:
    """Build VTK RAG corpus by chunking code examples and documentation.

    Processes raw VTK data files and produces chunked JSONL files ready
    for indexing.
    """

    def __init__(
        self,
        rag_client: RAGClient,
        mcp_client: VTKClient,
        base_path: Path | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            rag_client: RAG client providing config access.
            mcp_client: MCP client for VTK API access.
            base_path: Project root directory. Defaults to vtk_rag parent.
        """
        self.mcp_client = mcp_client
        self.base_path = base_path or Path(__file__).parent.parent.parent

        # Input paths
        self.examples_path = self.base_path / rag_client.raw_dir / rag_client.examples_file
        self.tests_path = self.base_path / rag_client.raw_dir / rag_client.tests_file
        self.docs_path = self.base_path / rag_client.raw_dir / rag_client.docs_file

        # Output paths
        self.code_output = self.base_path / rag_client.chunk_dir / rag_client.code_chunks
        self.docs_output = self.base_path / rag_client.chunk_dir / rag_client.doc_chunks

    def chunk(self) -> dict[str, Any]:
        """Chunk all sources (code and docs).

        Returns:
            Stats dict with keys: code, docs, total_chunks.
        """
        print("=" * 60)
        print("VTK RAG Corpus Builder")
        print("=" * 60)

        code_stats = self.chunk_code()
        doc_stats = self.chunk_docs()

        total_chunks = code_stats["chunks"] + doc_stats["chunks"]

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Code chunks:  {code_stats['chunks']:,}")
        print(f"  Doc chunks:   {doc_stats['chunks']:,}")
        print(f"  Total:        {total_chunks:,}")
        print("=" * 60)

        return {
            "code": code_stats,
            "docs": doc_stats,
            "total_chunks": total_chunks,
        }

    def chunk_code(self) -> dict[str, Any]:
        """Chunk code examples and tests.

        Reads from data/raw/vtk-python-examples.jsonl and vtk-python-tests.jsonl.
        Writes to data/processed/code-chunks.jsonl.

        Returns:
            Stats dict with keys: examples, tests, chunks, output.
        """
        self.code_output.parent.mkdir(parents=True, exist_ok=True)

        print("\n[1/2] Code Chunks")
        print("-" * 40)

        # Process examples
        print(f"  Examples: {self.examples_path.name}")
        examples_count, examples_chunks = self._process_code_file(
            self.examples_path, self.code_output, append=False
        )
        print(f"    → {examples_count} examples, {examples_chunks} chunks")

        # Process tests
        print(f"  Tests: {self.tests_path.name}")
        tests_count, tests_chunks = self._process_code_file(
            self.tests_path, self.code_output, append=True
        )
        print(f"    → {tests_count} tests, {tests_chunks} chunks")

        total_chunks = examples_chunks + tests_chunks
        print(f"  Output: {self.code_output}")
        print(f"  Total: {total_chunks} chunks")

        return {
            "examples": examples_count,
            "tests": tests_count,
            "chunks": total_chunks,
            "output": str(self.code_output),
        }

    def _process_code_file(
        self, input_path: Path, output_path: Path, append: bool = False
    ) -> tuple[int, int]:
        """Process a single code input file and write chunks to output."""
        mode = "a" if append else "w"
        total_examples = 0
        total_chunks = 0

        with input_path.open() as infile, output_path.open(mode) as outfile:
            for line in infile:
                if not line.strip():
                    continue

                record = json.loads(line)
                total_examples += 1

                code = record.get("code", "")
                if not code:
                    continue

                example_id = record.get("id", "unknown")
                chunker = CodeChunker(code, example_id, self.mcp_client)

                for chunk in chunker.extract_chunks():
                    outfile.write(json.dumps(chunk) + "\n")
                    total_chunks += 1

        return total_examples, total_chunks

    def chunk_docs(self) -> dict[str, Any]:
        """Chunk documentation.

        Reads from data/raw/vtk-python-docs.jsonl.
        Writes to data/processed/doc-chunks.jsonl.

        Returns:
            Stats dict with keys: docs, chunks, output.
        """
        print("\n[2/2] Documentation Chunks")
        print("-" * 40)
        print(f"  Input: {self.docs_path.name}")

        self.docs_output.parent.mkdir(parents=True, exist_ok=True)

        total_docs, total_doc_chunks, doc_chunk_type_counts = self._process_doc_file(
            self.docs_path, self.docs_output
        )

        print(f"    → {total_docs} docs, {total_doc_chunks} chunks")
        for doc_chunk_type, count in sorted(doc_chunk_type_counts.items()):
            print(f"      - {doc_chunk_type}: {count}")
        print(f"  Output: {self.docs_output}")

        return {
            "docs": total_docs,
            "chunks": total_doc_chunks,
            "output": str(self.docs_output),
        }

    def _process_doc_file(
        self, input_path: Path, output_path: Path
    ) -> tuple[int, int, dict[str, int]]:
        """Process a single doc input file and write chunks to output."""
        total_docs = 0
        total_chunks = 0
        chunk_type_counts: dict[str, int] = defaultdict(int)

        with input_path.open() as infile, output_path.open("w") as outfile:
            for line in infile:
                if not line.strip():
                    continue

                record = json.loads(line)
                total_docs += 1

                if not record.get("class_name"):
                    continue

                doc_chunker = DocChunker(record, self.mcp_client)

                for chunk in doc_chunker.extract_chunks():
                    outfile.write(json.dumps(chunk) + "\n")
                    chunk_type_counts[chunk.get("chunk_type", "unknown")] += 1
                    total_chunks += 1

        return total_docs, total_chunks, chunk_type_counts
