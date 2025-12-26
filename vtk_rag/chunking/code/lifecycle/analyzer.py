"""Analyze VTK object lifecycles from Python code.

This module provides the LifecycleAnalyzer class which extracts VTK object
lifecycles from function ASTs. It coordinates the visitor and builder to
produce lifecycle dictionaries.

Called by:
    CodeChunker in chunker.py

Code Map:
    LifecycleAnalyzer
        _analyze_vtk_lifecycles()  # main entry point
            └── LifecycleVisitor   # AST traversal (visitor.py)
            └── build_lifecycles() # lifecycle construction (builder.py)
"""

from __future__ import annotations

import ast

from vtk_rag.mcp import VTKClient

from .builder import build_lifecycles
from .models import LifecycleContext, VTKLifecycle
from .visitor import LifecycleVisitor


class LifecycleAnalyzer:
    """Extracts VTK object lifecycles from function ASTs."""

    def __init__(self, code: str, helper_methods: set[str],
                 function_defs: dict[str, ast.FunctionDef], mcp_client: VTKClient):
        """Initialize the lifecycle analyzer.

        Args:
            code: Source code being analyzed.
            helper_methods: Set of helper function names to exclude from lifecycle tracking.
            function_defs: Mapping of function names to their AST nodes.
            mcp_client: MCP client for VTK API access.
        """
        self.code = code
        self.helper_methods = helper_methods
        self.function_defs = function_defs
        self.mcp_client = mcp_client

    def _analyze_vtk_lifecycles(
        self, func_name: str, func_node: ast.FunctionDef, initial_var_to_class: dict[str, str] | None = None
    ) -> list[VTKLifecycle]:
        """Extract VTK class lifecycles from a function scope.

        Analyzes a single function (main, class method, or module-level code) to identify
        VTK object instantiations and their usage patterns. Does NOT analyze helper functions
        (those are chunked whole by HelperChunker).

        Args:
            func_name: Name of the function being analyzed (or "__module__" for module-level code)
            func_node: AST node for the function to analyze
            initial_var_to_class: Pre-built mapping of self.variables to VTK classes
                                 (used when analyzing class methods to track instance variables)

        Returns:
            List of lifecycle dictionaries, each containing:
                - variable: Variable name (or None for static methods)
                - class: VTK class name
                - type: Role classification (e.g., "properties", "infrastructure")
                - statements: AST nodes for all statements involving this variable
                - properties: Related property objects
                - mapper/actor: Relationship tracking for properties
        """
        # Create context to track all lifecycle data
        ctx = LifecycleContext()
        if initial_var_to_class:
            ctx.var_to_class = initial_var_to_class.copy()

        # Visit the AST to populate the context
        visitor = LifecycleVisitor(ctx, self.helper_methods, self.function_defs, self.mcp_client)
        visitor.visit(func_node)

        # Build lifecycle objects from the context
        return build_lifecycles(ctx, self.mcp_client)
