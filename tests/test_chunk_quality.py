"""Test chunk quality using LinearCellsDemo.py as a reference example.

This test verifies that the semantic chunker produces meaningful, coherent chunks
from a complex VTK example with multiple helper functions and cell types.
"""

from pathlib import Path

import pytest

from vtk_rag.chunking.code_chunker import CodeChunker


@pytest.fixture
def linear_cells_code() -> str:
    """Load the LinearCellsDemo.py fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "LinearCellsDemo.py"
    return fixture_path.read_text()


@pytest.fixture
def chunks(linear_cells_code: str) -> list[dict]:
    """Extract chunks from LinearCellsDemo.py."""
    chunker = CodeChunker(linear_cells_code, "https://examples.vtk.org/site/Python/LinearCellsDemo")
    return chunker.extract_chunks()


class TestChunkCount:
    """Test that chunking produces a reasonable number of chunks."""

    def test_produces_multiple_chunks(self, chunks: list[dict]):
        """Should produce multiple chunks from a 900+ line file."""
        assert len(chunks) > 10, f"Expected >10 chunks, got {len(chunks)}"

    def test_not_too_many_chunks(self, chunks: list[dict]):
        """Should not over-fragment the code."""
        assert len(chunks) < 100, f"Expected <100 chunks, got {len(chunks)}"


class TestHelperFunctionChunks:
    """Test that helper functions are chunked correctly."""

    def test_make_hexahedron_chunk_exists(self, chunks: list[dict]):
        """The make_hexahedron helper should produce a chunk."""
        hexahedron_chunks = [c for c in chunks if c.get("function_name") == "make_hexahedron"]
        assert len(hexahedron_chunks) >= 1, "Expected at least one chunk from make_hexahedron"

    def test_make_tetra_chunk_exists(self, chunks: list[dict]):
        """The make_tetra helper should produce a chunk."""
        tetra_chunks = [c for c in chunks if c.get("function_name") == "make_tetra"]
        assert len(tetra_chunks) >= 1, "Expected at least one chunk from make_tetra"

    def test_helper_chunks_contain_vtk_classes(self, chunks: list[dict]):
        """Helper function chunks should reference VTK classes."""
        helper_chunks = [c for c in chunks if c.get("function_name") not in ["main", "__module__", None]]

        for chunk in helper_chunks[:5]:  # Check first 5 helpers
            metadata = chunk.get("metadata", {})
            vtk_classes = metadata.get("vtk_classes", [])
            # At least some helper chunks should have VTK classes
            if vtk_classes:
                return  # Found one with VTK classes, test passes

        # If we get here, check if any helper has vtk_classes
        any_with_classes = any(
            c.get("metadata", {}).get("vtk_classes")
            for c in helper_chunks
        )
        assert any_with_classes, "Expected at least some helper chunks to have VTK classes"


class TestChunkMetadata:
    """Test that chunk metadata is populated correctly."""

    def test_all_chunks_have_chunk_id(self, chunks: list[dict]):
        """Every chunk should have a unique chunk_id."""
        chunk_ids = [c.get("chunk_id") for c in chunks]
        assert all(chunk_ids), "All chunks should have chunk_id"
        assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids should be unique"

    def test_all_chunks_have_content(self, chunks: list[dict]):
        """Every chunk should have non-empty content."""
        for chunk in chunks:
            content = chunk.get("content", "")
            assert content.strip(), f"Chunk {chunk.get('chunk_id')} has empty content"

    def test_all_chunks_have_type(self, chunks: list[dict]):
        """Every chunk should have a type."""
        for chunk in chunks:
            chunk_type = chunk.get("type")
            assert chunk_type, f"Chunk {chunk.get('chunk_id')} missing type"


class TestChunkContent:
    """Test that chunk content is coherent and self-contained."""

    def test_hexahedron_chunks_cover_function(self, chunks: list[dict]):
        """Chunks from make_hexahedron should cover key VTK classes used."""
        hexahedron_chunks = [c for c in chunks if c.get("function_name") == "make_hexahedron"]
        assert len(hexahedron_chunks) >= 1, "Expected chunks from make_hexahedron"

        # Combine all content from hexahedron chunks
        all_content = " ".join(c.get("content", "") for c in hexahedron_chunks)
        # Should have vtkPoints (used in the function)
        assert "vtkPoints" in all_content or "Points" in all_content

    def test_main_chunks_exist(self, chunks: list[dict]):
        """Should have chunks from the main function."""
        main_chunks = [c for c in chunks if c.get("function_name") == "main"]
        assert len(main_chunks) >= 1, "Expected at least one chunk from main()"

    def test_rendering_infrastructure_exists(self, chunks: list[dict]):
        """Should have rendering infrastructure chunks (renderer, window, etc.)."""
        rendering_types = ["Rendering Infrastructure", "vtkmodules.vtkRenderingCore"]
        rendering_chunks = [c for c in chunks if c.get("type") in rendering_types]
        # LinearCellsDemo has extensive rendering setup
        assert len(rendering_chunks) >= 1, "Expected rendering infrastructure chunks"


class TestChunkCoherence:
    """Test that chunks are semantically coherent."""

    def test_chunks_not_too_small(self, chunks: list[dict]):
        """Chunks should not be trivially small (< 50 chars)."""
        small_chunks = [c for c in chunks if len(c.get("content", "")) < 50]
        # Allow some small chunks but not too many
        ratio = len(small_chunks) / len(chunks) if chunks else 0
        assert ratio < 0.3, f"Too many small chunks: {len(small_chunks)}/{len(chunks)}"

    def test_chunks_not_too_large(self, chunks: list[dict]):
        """Chunks should not be excessively large (> 4000 chars ~1000 tokens)."""
        large_chunks = [c for c in chunks if len(c.get("content", "")) > 4000]
        assert len(large_chunks) < 3, f"Too many large chunks: {len(large_chunks)}"
