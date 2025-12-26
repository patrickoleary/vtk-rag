"""Extract semantic VTK chunks from Python code by analyzing object lifecycles.

Code Map:
    CodeChunker
        extract_chunks()                 # public API entry point
            ├── _extract_function_chunks()   # top-level functions (main, helpers)
            ├── _extract_module_chunks()     # module-level code
            └── _extract_class_chunks()      # class methods
                    ├── _build_class_var_map()       # self.var -> VTK class
                    └── _merge_class_lifecycles()    # merge across methods
"""

from __future__ import annotations

import ast
import warnings
from typing import Any

from vtk_rag.mcp import VTKClient

from .lifecycle import LifecycleAnalyzer
from .lifecycle.grouper import group_lifecycles
from .lifecycle.models import VTKLifecycle
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
        # Suppress SyntaxWarning from invalid escape sequences in VTK test code
        # (e.g., regex patterns with unescaped backslashes like \w, \()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
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

            grouped = group_lifecycles(lifecycles)

            # Pass helper_function name for non-main functions
            helper_function = None if func_name == 'main' else func_name

            for group in grouped:
                chunk = self.builder.build_chunk(group, helper_function)
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
            grouped = group_lifecycles(lifecycles)

            # Create chunks from grouped lifecycles
            for group in grouped:
                chunk = self.builder.build_chunk(group)
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
            grouped = group_lifecycles(merged_lifecycles)

            # Build chunks with class context
            for group in grouped:
                chunk = self.builder.build_chunk(group)
                chunks.append(chunk)

        return chunks

    # -------------------------------------------------------------------------
    # Class Lifecycle Helpers
    # Called by: _extract_class_chunks()
    # -------------------------------------------------------------------------

    def _merge_class_lifecycles(self, lifecycles: list[VTKLifecycle]) -> list[VTKLifecycle]:
        """Merge lifecycles for the same variable (e.g., self.mapper) across methods.

        When analyzing class methods, the same self.variable may appear in multiple methods.
        This merges them into a single lifecycle with combined statements and properties.
        """
        var_to_lifecycle: dict[str, VTKLifecycle] = {}

        for lc in lifecycles:
            # lc["variable"] is the variable name (e.g., "self.mapper")
            var_name = lc["variable"]
            # Skip static method calls (no variable)
            if var_name is None:
                continue

            # Strip "self." prefix for grouping (self.mapper → mapper)
            base_var = var_name.replace("self.", "")

            if base_var not in var_to_lifecycle:
                # First time seeing this variable — create new lifecycle entry
                var_to_lifecycle[base_var] = {
                    "variable": var_name,
                    "class": lc["class"],
                    "type": lc["type"],
                    "statements": lc["statements"][:],  # copy list
                    "properties": lc["properties"][:],
                    "mapper": lc.get("mapper"),
                    "actor": lc.get("actor"),
                    "methods": lc.get("methods", [])[:],
                    "method_calls": lc.get("method_calls", [])[:],
                }
            else:
                # Merge into existing lifecycle for this variable
                existing = var_to_lifecycle[base_var]
                existing["statements"].extend(lc["statements"])
                existing["properties"].extend(lc["properties"])
                existing["methods"].extend(lc.get("methods", []))
                existing["method_calls"].extend(lc.get("method_calls", []))
                # Update mapper/actor if not already set
                if not existing.get("mapper") and lc.get("mapper"):
                    existing["mapper"] = lc["mapper"]
                if not existing.get("actor") and lc.get("actor"):
                    existing["actor"] = lc["actor"]

        return list(var_to_lifecycle.values())

    def _build_class_var_map(self, class_node: ast.ClassDef) -> dict[str, str]:
        """Build a map of self.variable -> VTK class for all variables in a class.

        Walks all methods to find self.variable = vtkClass() assignments.
        Used as initial context when analyzing class methods.

        Args:
            class_node: AST node for the class definition.

        Returns:
            Dict mapping "self.varname" to VTK class name.
        """
        class_var_to_class: dict[str, str] = {}

        # 1. Iterate over class body items (methods, attributes, etc.)
        for item in class_node.body:
            # Only process method definitions
            if not isinstance(item, ast.FunctionDef):
                continue

            # 2. Walk all nodes in the method to find assignments
            for node in ast.walk(item):
                # Look for assignment statements
                if not isinstance(node, ast.Assign):
                    continue

                # 3. Check if RHS is a function call: self.var = something()
                #    AST: Assign(value=Call(...))
                if not isinstance(node.value, ast.Call):
                    continue

                # 4. Check if call is a simple name (not attribute): vtkClass()
                #    AST: Call(func=Name(id='vtkSphereSource'))
                if not isinstance(node.value.func, ast.Name):
                    continue

                # node.value.func.id is the class/function name being called
                vtk_class = node.value.func.id

                # 5. Check if it starts with "vtk" (potential VTK class)
                if not vtk_class.startswith("vtk"):
                    continue

                # 6. Verify it's a real VTK class via MCP
                if not self.mcp_client.is_vtk_class(vtk_class):
                    continue

                # 7. Check each assignment target for self.attribute pattern
                #    AST: Assign(targets=[Attribute(value=Name(id='self'), attr='mapper')])
                for target in node.targets:
                    # Must be an attribute access (not a simple name)
                    if not isinstance(target, ast.Attribute):
                        continue
                    # Must be on 'self' object
                    if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
                        continue
                    # target.attr is the attribute name (e.g., "mapper")
                    var_name = f"self.{target.attr}"
                    class_var_to_class[var_name] = vtk_class

        return class_var_to_class
