"""Build VTKLifecycle objects from LifecycleContext data.

This module transforms the raw tracking data collected by LifecycleVisitor into
structured VTKLifecycle objects. Each lifecycle represents a VTK object's usage
within a code scope, enriched with role, relationships, and method calls.

Called by:
    LifecycleAnalyzer._analyze_vtk_lifecycles() in analyzer.py

Code Map:
    build_lifecycles()
        ├── _build_instance_lifecycle()  # build lifecycle for a variable
        │       └── _collect_properties()    # gather explicit + inline properties
        └── _build_static_lifecycle()    # build lifecycle for static calls
"""

from __future__ import annotations

import ast

from vtk_rag.mcp import VTKClient

from .models import LifecycleContext, MethodCall, VTKLifecycle
from .utils import dedupe_method_calls, dedupe_preserve_order
from .vtk_knowledge import PROPERTY_MAPPINGS


def build_lifecycles(ctx: LifecycleContext, mcp_client: VTKClient) -> list[VTKLifecycle]:
    """Build VTKLifecycle objects from context data.

    Output: Instance lifecycles first, then static method call lifecycles.
    """
    lifecycles: list[VTKLifecycle] = []
    processed_vars: set[str] = set()

    # 1. Build lifecycles for instance variables (mapper, actor, etc.)
    for var_name, vtk_class in ctx.var_to_class.items():
        if var_name in processed_vars:
            continue  # already processed as a property of another variable

        if var_name in ctx.property_to_parent:
            continue  # skip properties, they're included with their parent

        # Build lifecycle for this variable
        lifecycle, prop_vars = _build_instance_lifecycle(var_name, vtk_class, ctx, mcp_client)
        lifecycles.append(lifecycle)

        # Mark variable and its properties as processed
        processed_vars.add(var_name)
        processed_vars.update(prop_vars)

    # 2. Build lifecycles for static method calls (vtkMath.Distance2BetweenPoints, etc.)
    for vtk_class, statements in ctx.static_method_calls.items():
        if statements:
            lifecycle = _build_static_lifecycle(vtk_class, statements, mcp_client)
            lifecycles.append(lifecycle)

    return lifecycles


def _build_instance_lifecycle(
    var_name: str,
    vtk_class: str,
    ctx: LifecycleContext,
    mcp_client: VTKClient
) -> tuple[VTKLifecycle, set[str]]:
    """Build a lifecycle for an instance variable.

    Returns lifecycle and set of property variable names (for marking as processed).
    """
    # 1. Collect statements and properties
    statements, properties, prop_vars = _collect_properties(var_name, vtk_class, ctx)

    # 2. Get VTK class role from MCP
    vtk_type = mcp_client.get_class_role(vtk_class) or "utility"

    # 3. Get mapper/actor relationships
    mapper_var = ctx.actor_to_mapper.get(var_name)
    actor_var = ctx.mapper_to_actor.get(var_name)

    # 4. Deduplicate methods and method_calls
    methods = dedupe_preserve_order(ctx.var_methods.get(var_name, []))
    method_calls = dedupe_method_calls(ctx.var_method_calls.get(var_name, []))

    # 5. Build lifecycle
    lifecycle: VTKLifecycle = {
        "variable": var_name,
        "class": vtk_class,
        "type": vtk_type,
        "statements": statements,
        "properties": properties,
        "mapper": mapper_var,
        "actor": actor_var,
        "methods": methods,
        "method_calls": method_calls,
    }

    return lifecycle, prop_vars


def _build_static_lifecycle(
    vtk_class: str,
    statements: list[ast.stmt],
    mcp_client: VTKClient
) -> VTKLifecycle:
    """Build a lifecycle for static method calls on a VTK class."""
    # 1. Get VTK class role and sort statements
    vtk_type = mcp_client.get_class_role(vtk_class) or "utility"
    sorted_stmts = sorted(statements, key=lambda s: s.lineno)

    # 2. Extract method names and arguments from AST
    methods: list[str] = []
    method_calls: list[MethodCall] = []
    for stmt in sorted_stmts:
        # Find call node in statement
        call_node = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value

        # Extract method name and arguments
        if call_node and isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr
            methods.append(method_name)
            # Extract arguments as strings
            args = []
            for arg in call_node.args:
                try:
                    args.append(ast.unparse(arg))
                except Exception:
                    args.append("...")
            method_calls.append({"name": method_name, "args": args})

    # 3. Deduplicate
    unique_methods = dedupe_preserve_order(methods)
    unique_method_calls = dedupe_method_calls(method_calls)

    # 4. Build lifecycle
    return {
        "variable": None,  # no variable for static calls
        "class": vtk_class,
        "type": vtk_type,
        "statements": sorted_stmts,
        "properties": [],
        "mapper": None,
        "actor": None,
        "methods": unique_methods,
        "method_calls": unique_method_calls,
    }


def _collect_properties(
    var_name: str,
    vtk_class: str,
    ctx: LifecycleContext
) -> tuple[list[ast.stmt], list[dict[str, str]], set[str]]:
    """Collect statements and properties for a variable.

    Returns (sorted_statements, properties_list, property_var_names).
    """
    statements = list(ctx.var_statements[var_name])
    properties: list[dict[str, str]] = []
    prop_vars: set[str] = set()

    # 1. Collect explicit property variables (prop = vtkProperty(); actor.SetProperty(prop))
    if var_name in ctx.parent_to_properties:
        for prop_var in ctx.parent_to_properties[var_name]:
            if prop_var in ctx.var_to_class:
                properties.append({
                    "variable": prop_var,
                    "class": ctx.var_to_class[prop_var],
                })
                statements.extend(ctx.var_statements[prop_var])
                prop_vars.add(prop_var)

    # 2. Add inline property marker for chained usage (actor.GetProperty().SetColor())
    if var_name in ctx.parent_has_chained_properties:
        # Infer property class from parent class
        prop_class = None
        for prop_cls, parent_classes in PROPERTY_MAPPINGS.items():
            if vtk_class in parent_classes:
                prop_class = prop_cls
                break
        if prop_class:
            properties.append({"variable": "inline", "class": prop_class})

    # 3. Sort statements by line number
    statements.sort(key=lambda s: s.lineno)

    return statements, properties, prop_vars
