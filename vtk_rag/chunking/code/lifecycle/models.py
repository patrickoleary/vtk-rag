"""Data models for VTK lifecycle analysis."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypedDict


@dataclass
class LifecycleContext:
    """Tracks VTK variable assignments, statements, and relationships during AST traversal.

    This context object is populated by LifecycleVisitor and consumed by build_lifecycles()
    to construct VTKLifecycle objects.
    """

    # Variable -> VTK class name
    var_to_class: dict[str, str] = field(default_factory=dict)

    # Variable -> list of AST statements
    var_statements: dict[str, list[ast.stmt]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Variable -> list of method names called
    var_methods: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Variable -> list of MethodCall dicts with args
    var_method_calls: dict[str, list[MethodCall]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Property variable -> parent actor variable
    property_to_parent: dict[str, str] = field(default_factory=dict)

    # Actor variable -> list of property variables
    parent_to_properties: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Variables with chained property usage (e.g., actor.GetProperty().SetColor())
    parent_has_chained_properties: set[str] = field(default_factory=set)

    # Mapper variable -> actor variable
    mapper_to_actor: dict[str, str] = field(default_factory=dict)

    # Actor variable -> mapper variable
    actor_to_mapper: dict[str, str] = field(default_factory=dict)

    # VTK class -> list of static method call statements
    static_method_calls: dict[str, list[ast.stmt]] = field(
        default_factory=lambda: defaultdict(list)
    )


class MethodCall(TypedDict):
    """Structure of a method call with its arguments."""
    name: str  # Method name (e.g., "SetRadius")
    args: list[str]  # String representations of arguments (e.g., ["10", "True"])


# -------------------------------------------------------------------------
# VTKLifecycle
#
# Represents a VTK object's usage within a code scope.
# Built by build_lifecycles() from LifecycleContext, grouped by group_lifecycles().
#
# Note: We use functional TypedDict syntax to allow 'class' as a key (Python keyword).
# -------------------------------------------------------------------------
VTKLifecycle = TypedDict('VTKLifecycle', {
    # Variable name (e.g., "actor", "self.mapper")
    # None for static method calls like vtkMath.Distance2BetweenPoints()
    'variable': str | None,

    # VTK class name (e.g., "vtkActor", "vtkPolyDataMapper")
    'class': str,

    # Role classification from MCP (e.g., "properties", "infrastructure", "data")
    'type': str,

    # AST statements belonging to this lifecycle, sorted by line number
    'statements': list[ast.stmt],

    # Related property objects with 'variable' and 'class' keys
    # Example: [{"variable": "prop", "class": "vtkProperty"}]
    'properties': list[dict[str, str]],

    # Mapper variable if this is an actor (e.g., actor.SetMapper(mapper))
    'mapper': str | None,

    # Actor variable if this is a mapper (e.g., actor.SetMapper(self))
    'actor': str | None,

    # Method names for simple filtering (e.g., "Start" in methods)
    'methods': list[str],

    # Method calls with arguments for detailed synopsis generation
    # Example: [{"name": "SetRadius", "args": ["1.0"]}]
    'method_calls': list[MethodCall],
}, total=False)


@dataclass
class LifecycleData:
    """Intermediate data collected from a lifecycle group.

    Used by SemanticChunk._collect_from_lifecycles() to pass data between
    helper methods when building a CodeChunk.
    """

    # AST statement nodes from all lifecycles in the group
    statements: list[ast.stmt]

    # VTK class names (e.g., "vtkSphereSource", "vtkActor")
    classes: list[str]

    # Variable names assigned to VTK objects
    variables: list[str]

    # Semantic role of the group (e.g., "properties", "infrastructure")
    role: str
