"""Extract semantic VTK chunks from Python code by analyzing object lifecycles."""

from __future__ import annotations

import ast
from typing import Any

from .lifecycle_analyzer import LifecycleAnalyzer
from .semantic_chunk_builder import SemanticChunkBuilder
from vtk_rag.mcp import VTK_API_CLIENT


class CodeChunker:
    """Extract semantic VTK chunks from Python code by analyzing object lifecycles."""

    def __init__(self, code: str, example_id: str) -> None:
        self.code = code
        self.example_id = example_id
        # Extract filename from example_id URL
        self.filename = example_id.split('/')[-2] if '/' in example_id else example_id
        self.chunk_counter = 0
        # Parse the code to access the AST.
        self.tree = ast.parse(code)
        # Build a map of function names to their AST nodes.
        self.function_defs: dict[str, ast.FunctionDef] = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        # Build a map of class names to their AST nodes.
        self.class_defs: dict[str, ast.ClassDef] = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
        }
        # Identify helper methods: all functions except 'main'
        self.helper_methods: set[str] = {name for name in self.function_defs.keys() if name != 'main'}
        # Build a map of helper function names to their return types by analyzing AST.
        self.helper_return_types: dict[str, str] = {}
        self._extract_helper_return_types()

        # Initialize analyzer and builder
        self.analyzer = LifecycleAnalyzer(code, self.helper_return_types,
                                         self.helper_methods, self.class_defs)
        self.builder = SemanticChunkBuilder(code, example_id, self.filename, self.class_defs)

    def _extract_helper_return_types(self) -> None:
        """Extract VTK return types from helper functions by analyzing their AST.

        Handles two patterns:
        1. Direct instantiation: return vtkClass()
        2. Variable return: return var (where var = vtkClass() earlier)
        """
        for func_name in self.helper_methods:
            func_node = self.function_defs[func_name]

            # Walk the function AST to find return statements
            for node in ast.walk(func_node):
                if isinstance(node, ast.Return) and node.value:
                    # Pattern 1: return vtkClass()
                    vtk_class = self._check_direct_instantiation(node.value)
                    if vtk_class:
                        self.helper_return_types[func_name] = vtk_class
                        break

                    # Pattern 2: return var (need to find var = vtkClass())
                    vtk_class = self._check_variable_return(node.value, func_node)
                    if vtk_class:
                        self.helper_return_types[func_name] = vtk_class
                        break

    def _check_direct_instantiation(self, return_value: ast.expr) -> str | None:
        """Check if return value is a direct VTK class instantiation.

        Pattern: return vtkClass() or return vtkClass(args...)
        """
        if isinstance(return_value, ast.Call):
            if isinstance(return_value.func, ast.Name):
                func_name = return_value.func.id
                if func_name.startswith("vtk"):
                    # Verify it's a real VTK class via MCP
                    resolved = VTK_API_CLIENT.resolve({func_name})
                    if func_name in resolved:
                        return func_name
        return None

    def _check_variable_return(self, return_value: ast.expr, func_node: ast.FunctionDef) -> str | None:
        """Check if return value is a variable assigned to a VTK class.

        Pattern: var = vtkClass(); return var
        """
        if isinstance(return_value, ast.Name):
            var_name = return_value.id
            # Find where this variable was assigned
            for stmt in ast.walk(func_node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            # Check if assigned value is a VTK class instantiation
                            vtk_class = self._check_direct_instantiation(stmt.value)
                            if vtk_class:
                                return vtk_class
        return None

    def extract_chunks(self) -> list[dict[str, Any]]:
        """Extract all semantic chunks from the code."""
        chunks = []

        # Process main function
        main_chunks = self._extract_main_chunks()
        chunks.extend(main_chunks)

        # Process helper functions - extract VTK lifecycles from helpers
        helper_chunks = self._extract_helper_chunks()
        chunks.extend(helper_chunks)

        # Process module-level code (code not inside any function)
        module_chunks = self._extract_module_chunks()
        chunks.extend(module_chunks)

        # Process class methods - extract VTK lifecycles with class context
        class_chunks = self._extract_class_chunks()
        chunks.extend(class_chunks)

        return chunks

    def _extract_main_chunks(self) -> list[dict[str, Any]]:
        """Extract lifecycle chunks from the main function."""
        chunks = []

        # Check if main function exists
        main_func_node = self.function_defs.get('main')
        if main_func_node:
            lifecycles = self.analyzer._analyze_vtk_lifecycles('main', main_func_node)

            # Group lifecycles by category
            grouped = self.analyzer._group_lifecycles(lifecycles)

            # Create chunks from grouped lifecycles
            for group in grouped:
                chunk = self.builder._build_chunk('main', group)
                chunks.append(chunk)

        return chunks

    def _extract_helper_chunks(self) -> list[dict[str, Any]]:
        """Extract lifecycle chunks from helper functions (all functions except main)."""
        chunks = []

        for helper_name in self.helper_methods:
            helper_func_node = self.function_defs.get(helper_name)
            if not helper_func_node:
                continue

            # Analyze VTK lifecycles in this helper function
            lifecycles = self.analyzer._analyze_vtk_lifecycles(helper_name, helper_func_node)

            if not lifecycles:
                # No VTK code in this helper - skip it (will be handled by helper_chunker if needed)
                continue

            # Group lifecycles by category
            grouped = self.analyzer._group_lifecycles(lifecycles)

            # Create chunks from grouped lifecycles
            for group in grouped:
                chunk = self.builder._build_chunk(helper_name, group)
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
                chunk = self.builder._build_chunk("__module__", group)
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
            class_var_to_class = self._build_class_var_map(class_node)

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
            merged_lifecycles = self._merge_class_lifecycles(all_lifecycles)

            if not merged_lifecycles:
                continue

            # Group the lifecycles
            grouped = self.analyzer._group_lifecycles(merged_lifecycles)

            # Build chunks with class context
            for group in grouped:
                chunk = self.builder._build_chunk(f"{class_name}.__merged__", group, is_class_chunk=True, class_name=class_name)
                chunks.append(chunk)

        return chunks

    def _build_class_var_map(self, class_node: ast.ClassDef) -> dict[str, str]:
        """Build a map of self.variable -> VTK class for all variables in the class."""
        class_var_to_class = {}

        # Walk all methods to find self.variable = vtkClass() assignments
        for item in class_node.body:
            if not isinstance(item, ast.FunctionDef):
                continue

            for node in ast.walk(item):
                if isinstance(node, ast.Assign):
                    # Check for self.variable = vtkClass()
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            vtk_class = node.value.func.id
                            if vtk_class.startswith("vtk"):
                                # Verify it's a real VTK class
                                resolved = VTK_API_CLIENT.resolve({vtk_class})
                                if vtk_class in resolved:
                                    for target in node.targets:
                                        if isinstance(target, ast.Attribute):
                                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                                var_name = f"self.{target.attr}"
                                                class_var_to_class[var_name] = vtk_class

        return class_var_to_class

    def _merge_class_lifecycles(self, lifecycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge lifecycles for the same variable (e.g., self.mapper) across methods."""
        # Group by variable name (strip self. for grouping)
        var_to_lifecycle: dict[str, dict[str, Any]] = {}

        for lc in lifecycles:
            var_name = lc["variable"]
            if var_name is None:  # Skip static methods
                continue

            # Strip self. for grouping
            base_var = var_name.replace("self.", "")

            if base_var not in var_to_lifecycle:
                # First time seeing this variable
                var_to_lifecycle[base_var] = {
                    "variable": var_name,
                    "class": lc["class"],
                    "type": lc["type"],
                    "statements": lc["statements"][:],
                    "properties": lc["properties"][:],
                    "mapper": lc.get("mapper"),
                    "actor": lc.get("actor"),
                }
            else:
                # Merge statements from this lifecycle
                existing = var_to_lifecycle[base_var]
                existing["statements"].extend(lc["statements"])
                existing["properties"].extend(lc["properties"])
                # Update mapper/actor if not set
                if not existing.get("mapper") and lc.get("mapper"):
                    existing["mapper"] = lc["mapper"]
                if not existing.get("actor") and lc.get("actor"):
                    existing["actor"] = lc["actor"]

        return list(var_to_lifecycle.values())
