"""Test chunk quality using LinearCellsDemo.py as a reference example.

This test verifies that the semantic chunker produces meaningful, coherent chunks
from a complex VTK example with multiple helper functions and cell types.
"""

from pathlib import Path

import pytest

from vtk_rag.chunking.code import CodeChunker


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing."""
    class DummyMCPClient:
        def is_vtk_class(self, class_name):
            return class_name.startswith("vtk")
        def get_class_modules(self, class_names):
            return {name: f"vtkmodules.vtk{name[3:6]}" for name in class_names if name.startswith("vtk")}
        def get_class_info(self, class_name):
            return {"role": "utility", "class_name": class_name}
        def get_class_role(self, class_name):
            # Return appropriate roles for common VTK classes
            if "Renderer" in class_name and "Window" not in class_name:
                return "renderer"
            if "RenderWindow" in class_name or "Interactor" in class_name:
                return "infrastructure"
            if "Camera" in class_name or "Light" in class_name:
                return "scene"
            if "Actor" in class_name or "Mapper" in class_name or "Property" in class_name:
                return "properties"
            if "Source" in class_name or "Reader" in class_name:
                return "input"
            if "Filter" in class_name:
                return "filter"
            return "utility"
        def get_class_visibility(self, class_name):
            return 0.8
        def get_class_action_phrase(self, class_name):
            return f"{class_name} usage"
        def get_class_input_datatype(self, class_name):
            return None
        def get_class_output_datatype(self, class_name):
            return None
        def get_method_signature(self, class_name, method_name):
            return None
        def get_class_synopsis(self, class_name):
            return None
        def get_class_doc(self, class_name):
            return None
    return DummyMCPClient()


@pytest.fixture
def linear_cells_code() -> str:
    """Load the LinearCellsDemo.py fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "LinearCellsDemo.py"
    return fixture_path.read_text()


@pytest.fixture
def chunks(linear_cells_code: str, mock_mcp_client) -> list[dict]:
    """Extract chunks from LinearCellsDemo.py."""
    chunker = CodeChunker(linear_cells_code, "https://examples.vtk.org/site/Python/LinearCellsDemo", mock_mcp_client)
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
    """Test that helper functions are chunked correctly and retrievable via semantic queries."""

    def test_make_hexahedron_retrievable_by_query(self, chunks: list[dict]):
        """Chunks for make_hexahedron should be retrievable via 'how to make a hexahedron' query."""
        # Find chunks that have the semantic query for hexahedron
        hexahedron_chunks = [
            c for c in chunks
            if any("how to make a hexahedron" in q for q in c.get("queries", []))
        ]
        assert len(hexahedron_chunks) >= 1, "Expected chunks retrievable via 'how to make a hexahedron' query"

    def test_make_tetra_retrievable_by_query(self, chunks: list[dict]):
        """Chunks for make_tetra should be retrievable via 'how to make a tetra' query."""
        tetra_chunks = [
            c for c in chunks
            if any("how to make a tetra" in q for q in c.get("queries", []))
        ]
        assert len(tetra_chunks) >= 1, "Expected chunks retrievable via 'how to make a tetra' query"

    def test_helper_chunks_contain_vtk_classes(self, chunks: list[dict]):
        """Helper function chunks (those with 'how to' queries) should reference VTK classes."""
        # Helper chunks have 'how to' queries generated from function names
        helper_chunks = [
            c for c in chunks
            if any(q.startswith("how to ") for q in c.get("queries", []))
        ]

        for chunk in helper_chunks[:5]:  # Check first 5 helpers
            vtk_classes = chunk.get("vtk_classes", [])
            # At least some helper chunks should have VTK classes
            if vtk_classes:
                return  # Found one with VTK classes, test passes

        # If we get here, check if any helper has vtk_classes
        any_with_classes = any(
            c.get("vtk_classes")
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

    def test_all_chunks_have_role(self, chunks: list[dict]):
        """Every chunk should have a role."""
        for chunk in chunks:
            role = chunk.get("role")
            assert role, f"Chunk {chunk.get('chunk_id')} missing role"


class TestChunkContent:
    """Test that chunk content is coherent and self-contained."""

    def test_hexahedron_chunks_cover_function(self, chunks: list[dict]):
        """Chunks retrievable via 'how to make a hexahedron' should cover key VTK classes used."""
        hexahedron_chunks = [
            c for c in chunks
            if any("how to make a hexahedron" in q for q in c.get("queries", []))
        ]
        assert len(hexahedron_chunks) >= 1, "Expected chunks retrievable via hexahedron query"

        # Combine all content from hexahedron chunks
        all_content = " ".join(c.get("content", "") for c in hexahedron_chunks)
        # Should have vtkPoints (used in the function)
        assert "vtkPoints" in all_content or "Points" in all_content

    def test_non_helper_chunks_exist(self, chunks: list[dict]):
        """Should have chunks from main/module code (those without 'how to' queries)."""
        # Non-helper chunks don't have 'how to' queries from function names
        non_helper_chunks = [
            c for c in chunks
            if not any(q.startswith("how to ") for q in c.get("queries", []))
        ]
        assert len(non_helper_chunks) >= 1, "Expected at least one chunk from main/module code"

    def test_rendering_infrastructure_exists(self, chunks: list[dict]):
        """Should have rendering infrastructure chunks (renderer, window, etc.)."""
        rendering_roles = ["renderer", "infrastructure", "scene"]
        rendering_chunks = [c for c in chunks if c.get("role") in rendering_roles]
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


class TestChunkGroupings:
    """Test that lifecycle groupings are correct for LinearCellsDemo.py."""

    def test_infrastructure_groups_window_and_interactor(self, chunks: list[dict]):
        """Infrastructure chunk should contain both vtkRenderWindow and vtkRenderWindowInteractor."""
        infra_chunks = [c for c in chunks if c.get("role") == "infrastructure"]
        assert len(infra_chunks) >= 1, "Expected at least one infrastructure chunk"

        # Find the chunk with both window and interactor
        for chunk in infra_chunks:
            vtk_classes = {cls["class"] for cls in chunk.get("vtk_classes", [])}
            if "vtkRenderWindow" in vtk_classes and "vtkRenderWindowInteractor" in vtk_classes:
                return  # Found it, test passes

        # If not found in same chunk, fail
        all_infra_classes = set()
        for chunk in infra_chunks:
            all_infra_classes.update(cls["class"] for cls in chunk.get("vtk_classes", []))
        pytest.fail(f"Expected vtkRenderWindow and vtkRenderWindowInteractor in same chunk, got: {all_infra_classes}")

    def test_infrastructure_chunk_ends_with_start(self, chunks: list[dict]):
        """Infrastructure chunk content should end with iren.Start() or similar."""
        infra_chunks = [c for c in chunks if c.get("role") == "infrastructure"]
        assert len(infra_chunks) >= 1, "Expected at least one infrastructure chunk"

        # Check that at least one infrastructure chunk ends with Start()
        for chunk in infra_chunks:
            content = chunk.get("content", "")
            if ".Start()" in content:
                # Verify Start() is near the end (last 100 chars)
                last_start_pos = content.rfind(".Start()")
                if last_start_pos > len(content) - 100:
                    return  # Found it at the end, test passes

        pytest.fail("Expected infrastructure chunk to end with .Start() call")

    def test_properties_groups_mapper_with_actor(self, chunks: list[dict]):
        """Properties chunks should group mappers with their actors."""
        props_chunks = [c for c in chunks if c.get("role") == "properties"]
        assert len(props_chunks) >= 1, "Expected at least one properties chunk"

        # Find chunks that have both a mapper and an actor
        mapper_actor_chunks = []
        for chunk in props_chunks:
            vtk_classes = {cls["class"] for cls in chunk.get("vtk_classes", [])}
            has_mapper = any("Mapper" in cls for cls in vtk_classes)
            has_actor = any("Actor" in cls for cls in vtk_classes)
            if has_mapper and has_actor:
                mapper_actor_chunks.append(chunk)

        assert len(mapper_actor_chunks) >= 1, "Expected at least one chunk with mapper+actor grouped together"

    def test_properties_groups_2d_mapper_with_actor2d(self, chunks: list[dict]):
        """Properties chunks should group vtkTextMapper/vtkLabeledDataMapper with vtkActor2D."""
        props_chunks = [c for c in chunks if c.get("role") == "properties"]

        # Find chunks with 2D mappers and Actor2D
        for chunk in props_chunks:
            vtk_classes = {cls["class"] for cls in chunk.get("vtk_classes", [])}
            has_2d_mapper = "vtkTextMapper" in vtk_classes or "vtkLabeledDataMapper" in vtk_classes
            has_actor2d = "vtkActor2D" in vtk_classes
            if has_2d_mapper and has_actor2d:
                return  # Found it, test passes

        pytest.fail("Expected vtkTextMapper or vtkLabeledDataMapper grouped with vtkActor2D")

    def test_infrastructure_does_not_contain_loop_body(self, chunks: list[dict]):
        """Infrastructure chunk should not contain unrelated loop body code like mappers/actors."""
        infra_chunks = [c for c in chunks if c.get("role") == "infrastructure"]

        for chunk in infra_chunks:
            content = chunk.get("content", "")
            # These should NOT be in infrastructure chunks
            assert "vtkDataSetMapper" not in content, "Infrastructure chunk should not contain vtkDataSetMapper"
            assert "vtkGlyph3DMapper" not in content, "Infrastructure chunk should not contain vtkGlyph3DMapper"
            assert "label_mapper" not in content, "Infrastructure chunk should not contain label_mapper"
