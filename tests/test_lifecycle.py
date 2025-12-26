"""Tests for the lifecycle analysis modules."""

import ast
from unittest.mock import MagicMock

import pytest


class TestLifecycleModels:
    """Tests for lifecycle data models."""

    def test_lifecycle_context_import(self):
        """Test that LifecycleContext can be imported."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext
        assert LifecycleContext is not None

    def test_lifecycle_context_defaults(self):
        """Test LifecycleContext default values."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext
        ctx = LifecycleContext()
        assert ctx.var_to_class == {}
        assert len(ctx.var_statements) == 0
        assert len(ctx.var_methods) == 0
        assert len(ctx.var_method_calls) == 0
        assert ctx.property_to_parent == {}
        assert len(ctx.parent_to_properties) == 0
        assert ctx.parent_has_chained_properties == set()
        assert ctx.mapper_to_actor == {}
        assert ctx.actor_to_mapper == {}
        assert len(ctx.static_method_calls) == 0

    def test_method_call_typed_dict(self):
        """Test MethodCall TypedDict structure."""
        from vtk_rag.chunking.code.lifecycle.models import MethodCall
        mc: MethodCall = {"name": "SetRadius", "args": ["1.0"]}
        assert mc["name"] == "SetRadius"
        assert mc["args"] == ["1.0"]

    def test_vtk_lifecycle_typed_dict(self):
        """Test VTKLifecycle TypedDict structure."""
        from vtk_rag.chunking.code.lifecycle.models import VTKLifecycle
        lifecycle: VTKLifecycle = {
            "variable": "sphere",
            "class": "vtkSphereSource",
            "type": "source_geometric",
            "statements": [],
            "properties": [],
            "mapper": None,
            "actor": None,
            "methods": ["SetRadius"],
            "method_calls": [{"name": "SetRadius", "args": ["1.0"]}],
        }
        assert lifecycle["variable"] == "sphere"
        assert lifecycle["class"] == "vtkSphereSource"

    def test_lifecycle_data_import(self):
        """Test that LifecycleData can be imported."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleData
        assert LifecycleData is not None


class TestLifecycleUtils:
    """Tests for lifecycle utility functions."""

    def test_dedupe_preserve_order(self):
        """Test dedupe_preserve_order maintains order."""
        from vtk_rag.chunking.code.lifecycle.utils import dedupe_preserve_order
        result = dedupe_preserve_order([1, 2, 3, 2, 1, 4])
        assert result == [1, 2, 3, 4]

    def test_dedupe_preserve_order_empty(self):
        """Test dedupe_preserve_order with empty list."""
        from vtk_rag.chunking.code.lifecycle.utils import dedupe_preserve_order
        result = dedupe_preserve_order([])
        assert result == []

    def test_dedupe_preserve_order_strings(self):
        """Test dedupe_preserve_order with strings."""
        from vtk_rag.chunking.code.lifecycle.utils import dedupe_preserve_order
        result = dedupe_preserve_order(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_dedupe_method_calls(self):
        """Test dedupe_method_calls keeps first occurrence."""
        from vtk_rag.chunking.code.lifecycle.utils import dedupe_method_calls
        calls = [
            {"name": "SetRadius", "args": ["1.0"]},
            {"name": "SetCenter", "args": ["0", "0", "0"]},
            {"name": "SetRadius", "args": ["2.0"]},  # Duplicate name
        ]
        result = dedupe_method_calls(calls)
        assert len(result) == 2
        assert result[0]["name"] == "SetRadius"
        assert result[0]["args"] == ["1.0"]  # First occurrence kept
        assert result[1]["name"] == "SetCenter"

    def test_dedupe_method_calls_empty(self):
        """Test dedupe_method_calls with empty list."""
        from vtk_rag.chunking.code.lifecycle.utils import dedupe_method_calls
        result = dedupe_method_calls([])
        assert result == []


class TestLifecycleAnalyzer:
    """Tests for LifecycleAnalyzer class."""

    def test_import(self):
        """Test that LifecycleAnalyzer can be imported."""
        from vtk_rag.chunking.code.lifecycle import LifecycleAnalyzer
        assert LifecycleAnalyzer is not None

    def test_init(self, mock_mcp_client):
        """Test LifecycleAnalyzer initialization."""
        from vtk_rag.chunking.code.lifecycle import LifecycleAnalyzer
        code = "sphere = vtk.vtkSphereSource()"
        helper_methods: set[str] = set()
        function_defs: dict = {}
        analyzer = LifecycleAnalyzer(code, helper_methods, function_defs, mock_mcp_client)
        assert analyzer.code == code
        assert analyzer.mcp_client == mock_mcp_client


class TestLifecycleVisitor:
    """Tests for LifecycleVisitor class."""

    def test_import(self):
        """Test that LifecycleVisitor can be imported."""
        from vtk_rag.chunking.code.lifecycle.visitor import LifecycleVisitor
        assert LifecycleVisitor is not None

    def test_visit_vtk_assignment(self, mock_mcp_client):
        """Test visitor detects VTK class assignment."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext
        from vtk_rag.chunking.code.lifecycle.visitor import LifecycleVisitor

        code = "sphere = vtk.vtkSphereSource()"
        tree = ast.parse(code)
        ctx = LifecycleContext()
        helper_methods: set[str] = set()
        function_defs: dict = {}
        visitor = LifecycleVisitor(ctx, helper_methods, function_defs, mock_mcp_client)
        visitor.visit(tree)

        assert "sphere" in ctx.var_to_class
        assert ctx.var_to_class["sphere"] == "vtkSphereSource"

    def test_visit_method_call(self, mock_mcp_client):
        """Test visitor tracks method calls."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext
        from vtk_rag.chunking.code.lifecycle.visitor import LifecycleVisitor

        code = """
sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.0)
"""
        tree = ast.parse(code)
        ctx = LifecycleContext()
        helper_methods: set[str] = set()
        function_defs: dict = {}
        visitor = LifecycleVisitor(ctx, helper_methods, function_defs, mock_mcp_client)
        visitor.visit(tree)

        assert "sphere" in ctx.var_methods
        assert "SetRadius" in ctx.var_methods["sphere"]

    def test_visit_pythonic_property(self, mock_mcp_client):
        """Test visitor tracks Pythonic property access as SetX call."""
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext
        from vtk_rag.chunking.code.lifecycle.visitor import LifecycleVisitor

        code = """
sphere = vtk.vtkSphereSource()
sphere.radius = 1.0
"""
        tree = ast.parse(code)
        ctx = LifecycleContext()
        helper_methods: set[str] = set()
        function_defs: dict = {}
        visitor = LifecycleVisitor(ctx, helper_methods, function_defs, mock_mcp_client)
        visitor.visit(tree)

        assert "sphere" in ctx.var_methods
        # Pythonic 'radius' should be tracked as 'SetRadius'
        assert "SetRadius" in ctx.var_methods["sphere"]


class TestLifecycleBuilder:
    """Tests for lifecycle builder functions."""

    def test_build_lifecycles_import(self):
        """Test that build_lifecycles can be imported."""
        from vtk_rag.chunking.code.lifecycle.builder import build_lifecycles
        assert build_lifecycles is not None

    def test_build_lifecycles_simple(self, mock_mcp_client):
        """Test building lifecycles from context."""
        from vtk_rag.chunking.code.lifecycle.builder import build_lifecycles
        from vtk_rag.chunking.code.lifecycle.models import LifecycleContext

        code = "sphere = vtk.vtkSphereSource()"
        tree = ast.parse(code)
        ctx = LifecycleContext()
        ctx.var_to_class["sphere"] = "vtkSphereSource"
        ctx.var_statements["sphere"] = [tree.body[0]]
        ctx.var_methods["sphere"] = ["SetRadius"]
        ctx.var_method_calls["sphere"] = [{"name": "SetRadius", "args": ["1.0"]}]

        lifecycles = build_lifecycles(ctx, mock_mcp_client)

        assert len(lifecycles) == 1
        assert lifecycles[0]["variable"] == "sphere"
        assert lifecycles[0]["class"] == "vtkSphereSource"


class TestLifecycleGrouper:
    """Tests for lifecycle grouping functions."""

    def test_group_lifecycles_import(self):
        """Test that group_lifecycles can be imported."""
        from vtk_rag.chunking.code.lifecycle.grouper import group_lifecycles
        assert group_lifecycles is not None

    def test_group_lifecycles_empty(self):
        """Test grouping empty lifecycles list."""
        from vtk_rag.chunking.code.lifecycle.grouper import group_lifecycles
        groups = group_lifecycles([])
        assert groups == []

    def test_group_lifecycles_single(self, mock_mcp_client):
        """Test grouping a single lifecycle."""
        from vtk_rag.chunking.code.lifecycle.grouper import group_lifecycles

        lifecycle = {
            "variable": "sphere",
            "class": "vtkSphereSource",
            "type": "source_geometric",
            "statements": [],
            "properties": [],
            "mapper": None,
            "actor": None,
            "methods": [],
            "method_calls": [],
        }

        groups = group_lifecycles([lifecycle])
        assert len(groups) >= 1


class TestVTKKnowledge:
    """Tests for VTK knowledge constants."""

    def test_property_mappings_import(self):
        """Test that PROPERTY_MAPPINGS can be imported."""
        from vtk_rag.chunking.code.lifecycle.vtk_knowledge import PROPERTY_MAPPINGS
        assert isinstance(PROPERTY_MAPPINGS, dict)
        assert "vtkProperty" in PROPERTY_MAPPINGS

    def test_actor_like_props_import(self):
        """Test that ACTOR_LIKE_PROPS can be imported."""
        from vtk_rag.chunking.code.lifecycle.vtk_knowledge import ACTOR_LIKE_PROPS
        assert isinstance(ACTOR_LIKE_PROPS, set)
        # vtkVolume is an actor-like prop (not vtkActor itself)
        assert "vtkVolume" in ACTOR_LIKE_PROPS

    def test_property_getters_import(self):
        """Test that PROPERTY_GETTERS can be imported."""
        from vtk_rag.chunking.code.lifecycle.vtk_knowledge import PROPERTY_GETTERS
        assert isinstance(PROPERTY_GETTERS, set)
        assert "GetProperty" in PROPERTY_GETTERS

    def test_chainable_getters_import(self):
        """Test that CHAINABLE_GETTERS can be imported."""
        from vtk_rag.chunking.code.lifecycle.vtk_knowledge import CHAINABLE_GETTERS
        assert isinstance(CHAINABLE_GETTERS, set)


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing."""
    client = MagicMock()
    client.is_vtk_class.side_effect = lambda name: name.startswith("vtk")
    client.get_class_role.return_value = "utility"
    client.get_class_visibility.return_value = 0.5
    client.get_class_action_phrase.return_value = None
    client.get_class_synopsis.return_value = None
    client.get_class_input_datatype.return_value = None
    client.get_class_output_datatype.return_value = None
    return client
