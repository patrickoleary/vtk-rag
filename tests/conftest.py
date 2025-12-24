"""Pytest configuration and fixtures for VTK RAG tests."""

from pathlib import Path

import pytest


@pytest.fixture
def base_path() -> Path:
    """Return the project base path."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(base_path: Path) -> Path:
    """Return the data directory path."""
    return base_path / "data"


@pytest.fixture
def raw_dir(data_dir: Path) -> Path:
    """Return the raw data directory path."""
    return data_dir / "raw"


@pytest.fixture
def chunk_dir(data_dir: Path) -> Path:
    """Return the processed data directory path."""
    return data_dir / "processed"


@pytest.fixture
def sample_code_chunk() -> dict:
    """Return a sample code chunk for testing."""
    return {
        "chunk_id": "test-chunk-001",
        "example_id": "https://examples.vtk.org/site/Python/GeometricObjects/Sphere",
        "synopsis": "Sphere source with radius set to 1.0, center set to (0, 0, 0)",
        "content": "sphere = vtkSphereSource()\nsphere.SetRadius(1.0)\nsphere.SetCenter(0, 0, 0)",
        "role": "input",
        "visibility_score": 0.9,
        "vtk_class": "vtkSphereSource",
        "variable_name": "sphere",
        "input_datatype": "",
        "output_datatype": "vtkPolyData",
        "action_phrase": "create a sphere source",
        "queries": ["How to create a sphere in VTK?", "vtkSphereSource example"],
        "metadata": {
            "vtk_classes": [{"class": "vtkSphereSource", "variable": "sphere"}],
        },
    }


@pytest.fixture
def sample_doc_chunk() -> dict:
    """Return a sample doc chunk for testing."""
    return {
        "chunk_id": "doc-vtkSphereSource-overview",
        "chunk_type": "class_overview",
        "vtk_class": "vtkSphereSource",
        "content": "vtkSphereSource creates a sphere centered at origin.",
        "synopsis": "Create a sphere (default representation).",
        "role": "input",
        "action_phrase": "create a sphere",
        "visibility_score": 0.9,
        "module": "vtkmodules.vtkFiltersSources",
        "input_datatype": "",
        "output_datatype": "vtkPolyData",
        "queries": ["What is vtkSphereSource?", "How to create a sphere?"],
    }
