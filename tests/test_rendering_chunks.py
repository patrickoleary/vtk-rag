"""Regression tests for rendering chunk separation."""

from __future__ import annotations

import pytest

from vtk_rag.chunking.code import CodeChunker
from vtk_rag.chunking.code.lifecycle.grouper import _group_infrastructure, group_lifecycles


@pytest.fixture
def mock_mcp_client():
    """Stub the MCP client so tests don't hit external services."""

    class DummyClient:
        class_map = {
            "vtkCamera": "vtkmodules.vtkRenderingCore",
            "vtkRenderer": "vtkmodules.vtkRenderingCore",
            "vtkRenderWindow": "vtkmodules.vtkRenderingOpenGL2",
            "vtkRenderWindowInteractor": "vtkmodules.vtkRenderingUI",
        }
        roles = {
            "vtkCamera": "scene",
            "vtkRenderer": "renderer",
            "vtkRenderWindow": "infrastructure",
            "vtkRenderWindowInteractor": "infrastructure",
        }

        def is_vtk_class(self, class_name: str) -> bool:
            return class_name in self.class_map

        def get_class_modules(self, class_names: set[str]) -> dict[str, str]:
            return {
                name: self.class_map.get(name, "vtkmodules.vtkRenderingCore")
                for name in class_names
                if name in self.class_map
            }

        def get_class_role(self, class_name: str) -> str | None:
            return self.roles.get(class_name, "other")

        def get_class_info(self, class_name: str) -> dict | None:
            role = self.roles.get(class_name)
            if role:
                return {"role": role, "class_name": class_name}
            return None

        def get_class_visibility(self, class_name: str) -> float | None:
            return 0.8

        def get_class_action_phrase(self, class_name: str) -> str | None:
            return f"{class_name} usage"

        def get_class_input_datatype(self, class_name: str) -> str | None:
            return None

        def get_class_output_datatype(self, class_name: str) -> str | None:
            return None

        def get_method_info(self, class_name: str, method_name: str) -> dict | None:
            return None

        def get_method_signature(self, class_name: str, method_name: str) -> str | None:
            return None

    return DummyClient()


def _base_lifecycle(vtk_class: str, var: str, role: str) -> dict:
    """Helper to build a minimal lifecycle dict for tests."""
    return {
        "variable": var,
        "class": vtk_class,
        "type": role,
        "statements": [],
        "properties": [],
        "mapper": None,
        "actor": None,
        "methods": [],
        "method_calls": [],
    }


def test_group_lifecycles_groups_infrastructure_and_singles(mock_mcp_client):
    """LifecycleAnalyzer groups infrastructure together, other roles are single-lifecycle."""
    # LifecycleAnalyzer not needed for group_lifecycles test
    lifecycles = [
        _base_lifecycle("vtkCamera", "camera", "scene"),
        _base_lifecycle("vtkRenderer", "renderer", "renderer"),
        _base_lifecycle("vtkRenderWindow", "ren_win", "infrastructure"),
        _base_lifecycle("vtkRenderWindowInteractor", "iren", "infrastructure"),
    ]

    groups = group_lifecycles(lifecycles)

    # Infrastructure grouped together, scene and renderer are single-lifecycle groups
    assert len(groups) == 3, "Expected infrastructure group + 2 single-lifecycle groups"

    # First group is infrastructure (sorted: window before interactor)
    infrastructure_group_classes = [lc["class"] for lc in groups[0]]
    assert infrastructure_group_classes == ["vtkRenderWindow", "vtkRenderWindowInteractor"]

    # Remaining groups are single-lifecycle (scene, renderer)
    single_classes = {groups[1][0]["class"], groups[2][0]["class"]}
    assert single_classes == {"vtkCamera", "vtkRenderer"}


def test_group_infrastructure_splits_start_after_styles(mock_mcp_client):
    """Infrastructure grouping should split interactor: setup, then styles, then Start() last."""
    import ast

    # LifecycleAnalyzer not needed - testing _group_infrastructure directly
    _ = mock_mcp_client  # Silence unused fixture warning

    # Create AST statements for the interactor
    setup_stmt = ast.parse("iren.SetRenderWindow(window)").body[0]
    start_stmt = ast.parse("iren.Start()").body[0]

    # Interactor lifecycle with Start() in methods and statements
    interactor_lc = {
        "variable": "iren",
        "class": "vtkRenderWindowInteractor",
        "type": "infrastructure",
        "statements": [setup_stmt, start_stmt],
        "properties": [],
        "mapper": None,
        "actor": None,
        "methods": ["SetRenderWindow", "Start"],
        "method_calls": [],
    }

    # InteractorStyle lifecycle
    style_lc = _base_lifecycle("vtkInteractorStyleTrackballCamera", "style", "infrastructure")

    # Window lifecycle
    window_lc = _base_lifecycle("vtkRenderWindow", "window", "infrastructure")

    lifecycles = [interactor_lc, style_lc, window_lc]

    groups = _group_infrastructure(lifecycles)

    # Should return one group with all infrastructure
    assert len(groups) == 1
    infra_group = groups[0]

    # Order should be: window, interactor (without Start), style, Start lifecycle
    assert len(infra_group) == 4, "Expected window, interactor, style, start"

    assert infra_group[0]["class"] == "vtkRenderWindow"
    assert infra_group[1]["class"] == "vtkRenderWindowInteractor"
    assert "Start" not in infra_group[1].get("methods", [])
    assert infra_group[2]["class"] == "vtkInteractorStyleTrackballCamera"
    assert infra_group[3]["class"] == "vtkRenderWindowInteractor"
    assert infra_group[3].get("methods") == ["Start"]


def test_code_chunker_emits_renderer_and_window_chunks(mock_mcp_client):
    """End-to-end chunking should produce distinct renderer vs window chunks."""
    code = """
import vtk

def main():
    renderer = vtk.vtkRenderer()
    camera = vtk.vtkCamera()
    renderer.SetActiveCamera(camera)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
"""

    chunker = CodeChunker(code, example_id="tests/rendering_example", mcp_client=mock_mcp_client)
    chunks = chunker.extract_chunks()

    scene_chunk = next((c for c in chunks if c["role"] == "scene"), None)
    renderer_chunk = next((c for c in chunks if c["role"] == "renderer"), None)
    infrastructure_chunk = next((c for c in chunks if c["role"] == "infrastructure"), None)

    assert scene_chunk is not None, "Expected a scene chunk"
    assert renderer_chunk is not None, "Expected a renderer chunk"
    assert infrastructure_chunk is not None, "Expected an infrastructure chunk"

    scene_classes = {cls["class"] for cls in scene_chunk["vtk_classes"]}
    renderer_classes = {cls["class"] for cls in renderer_chunk["vtk_classes"]}
    infrastructure_classes = {cls["class"] for cls in infrastructure_chunk["vtk_classes"]}

    assert "vtkCamera" in scene_classes
    assert "vtkRenderer" in renderer_classes
    assert "vtkRenderWindow" in infrastructure_classes
    assert "vtkRenderWindowInteractor" in infrastructure_classes
