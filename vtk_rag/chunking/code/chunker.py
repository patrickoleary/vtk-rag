"""Extract semantic VTK chunks from Python code by analyzing object lifecycles."""

from __future__ import annotations

import ast
from typing import Any

from vtk_rag.mcp import VTKClient

from .lifecycle_analyzer import LifecycleAnalyzer
from .semantic_chunk import SemanticChunk


class CodeChunker:
    """Extract semantic VTK chunks from Python code by analyzing object lifecycles."""

    def __init__(self, code: str, example_id: str, mcp_client: VTKClient) -> None:
        self.code = code
        self.example_id = example_id
        self.mcp_client = mcp_client
        # Extract filename from example_id URL
        self.filename = example_id.split('/')[-2] if '/' in example_id else example_id
        # Parse the code to access the AST.
        self.tree = ast.parse(code)
        # Build a map of function names to their AST nodes.
        self.function_defs: dict[str, ast.FunctionDef] = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        # Identify helper methods: all functions except 'main'
        self.helper_methods: set[str] = {name for name in self.function_defs.keys() if name != 'main'}

        # Initialize analyzer and builder
        self.analyzer = LifecycleAnalyzer(code, self.helper_methods,
                                         self.function_defs, mcp_client)
        self.builder = SemanticChunk(code, example_id, self.filename, mcp_client)

    def extract_chunks(self) -> list[dict[str, Any]]:
        """Extract all semantic chunks from the code."""
        chunks = []

        # Extract VTK lifecycles from top-level functions (main and helpers)
        chunks.extend(self._extract_function_chunks())

        # Extract VTK lifecycles from module-level code
        chunks.extend(self._extract_module_chunks())

        # Extract VTK lifecycles from class methods
        chunks.extend(self._extract_class_chunks())

        return chunks

    def _extract_function_chunks(self) -> list[dict[str, Any]]:
        """Extract lifecycle chunks from all top-level functions (main and helpers)."""
        chunks = []

        for func_name, func_node in self.function_defs.items():
            lifecycles = self.analyzer._analyze_vtk_lifecycles(func_name, func_node)

            if not lifecycles:
                continue

            grouped = self.analyzer._group_lifecycles(lifecycles)

            # Pass helper_function name for non-main functions
            helper_function = None if func_name == 'main' else func_name

            for group in grouped:
                chunk = self.builder._build_chunk(group, helper_function)
                chunks.append(chunk)

        return chunks

    def _extract_module_chunks(self) -> list[dict[str, Any]]:
        """Extract lifecycle chunks from module-level code (code not inside any function)."""
        chunks = []

        # Filter out function definitions and class definitions
        module_statements = [
            node for node in self.tree.body
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        ]

        if not module_statements:
            return chunks

        # Create a synthetic function node
        synthetic_func = ast.FunctionDef(
            name="__module__",
            args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=module_statements,
            decorator_list=[],
            returns=None,
        )

        # Analyze module-level code
        lifecycles = self.analyzer._analyze_vtk_lifecycles("__module__", synthetic_func)

        if lifecycles:
            # Group lifecycles by category
            grouped = self.analyzer._group_lifecycles(lifecycles)

            # Create chunks from grouped lifecycles
            for group in grouped:
                chunk = self.builder._build_chunk(group)
                chunks.append(chunk)

        return chunks

    def _extract_class_chunks(self) -> list[dict[str, Any]]:
        """Extract semantic chunks from user-defined classes that contain VTK code.
        """
        chunks = []

        for class_node in self.tree.body:
            if not isinstance(class_node, ast.ClassDef):
                continue

            class_name = class_node.name

            # First pass: build class-level var_to_class for all self.variables
            class_var_to_class = self.analyzer._build_class_var_map(class_node)

            if not class_var_to_class:
                continue

            # Second pass: analyze all methods with class-level variable knowledge
            all_lifecycles = []

            for item in class_node.body:
                if isinstance(item, ast.FunctionDef):
                    method_lifecycles = self.analyzer._analyze_vtk_lifecycles(
                        f"{class_name}.{item.name}", item, class_var_to_class
                    )
                    all_lifecycles.extend(method_lifecycles)

            if not all_lifecycles:
                continue

            # Merge lifecycles for the same self.variable across methods
            merged_lifecycles = self.analyzer._merge_class_lifecycles(all_lifecycles)

            if not merged_lifecycles:
                continue

            # Group the lifecycles
            grouped = self.analyzer._group_lifecycles(merged_lifecycles)

            # Build chunks with class context
            for group in grouped:
                chunk = self.builder._build_chunk(group)
                chunks.append(chunk)

        return chunks

