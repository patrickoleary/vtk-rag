"""AST visitor for extracting VTK lifecycle information from Python code.

This module provides LifecycleVisitor which walks function ASTs to track VTK
object instantiations, method calls, and relationships (mapper/actor, property/parent).

Called by:
    LifecycleAnalyzer._analyze_vtk_lifecycles() in analyzer.py

Code Map:
    LifecycleVisitor
        visit_Assign()                    # assignment statements
            ├── _handle_call_assignment()
            │       ├── _handle_attribute_call_assignment()  # obj.Method()
            │       │       └── _handle_property_getter_assignment()
            │       └── _handle_name_call_assignment()       # vtkClass() or helper()
            └── _handle_chained_call_assignment()            # obj.Get*().Method()
        visit_Expr()                      # expression statements
            └── _handle_method_call()
                    ├── _track_method_call()
                    └── _track_relationship_from_method()
        visit_For()                       # for-loop statements
            ├── _collect_vtk_vars_per_statement()
            │       └── _extract_vtk_var_from_node()
            └── _is_single_var_loop()
        _extract_helper_return_types()    # helper function analysis
"""

from __future__ import annotations

import ast

from vtk_rag.mcp import VTKClient

from .models import LifecycleContext
from .vtk_knowledge import (
    ACTOR_LIKE_PROPS,
    CHAINABLE_GETTERS,
    PROPERTY_GETTERS,
    PROPERTY_SETTERS,
)


class LifecycleVisitor(ast.NodeVisitor):
    """AST visitor that populates a LifecycleContext with VTK object usage information.

    Tracks VTK class instantiations, method calls, and relationships between
    objects (mapper/actor, property/parent).
    """

    def __init__(
        self,
        ctx: LifecycleContext,
        helper_methods: set[str],
        function_defs: dict[str, ast.FunctionDef],
        mcp_client: VTKClient,
    ) -> None:
        """Initialize the visitor.

        Args:
            ctx: Context object to populate with lifecycle data.
            helper_methods: Set of helper function names to exclude from tracking.
            function_defs: Mapping of function names to their AST nodes.
            mcp_client: MCP client for VTK API access.
        """
        self.ctx = ctx
        self.helper_methods = helper_methods
        self.function_defs = function_defs
        self.mcp_client = mcp_client
        # Extract helper return types from AST (computed internally)
        self.helper_return_types = self._extract_helper_return_types()
        self._current_statement: ast.stmt | None = None

    # =========================================================================
    # visit_Assign - Core Visitor Method (called by ast.NodeVisitor.visit())
    # =========================================================================
    #
    # Assignment Patterns Handled:
    #
    # 1. Call assignments (_handle_call_assignment):
    #    a. Attribute calls (_handle_attribute_call_assignment): (have a dot '.')
    #       - Module instantiation: var = vtk.vtkSphereSource()
    #       - Static method call: result = vtkMath.Distance2BetweenPoints(p1, p2)
    #    b. Direct calls (_handle_direct_call_assignment): (no dot '.')
    #       - VTK class instantiation: sphere = vtkSphereSource()
    #       - VTK Pythonic constructor keywords (_track_pythonic_constructor_keywords):
    #         sphere = vtkSphereSource(radius=1.0, center=(0,0,0))
    #         actor = vtkActor(mapper=m, property=p, position=(1,2,3))
    #         (each keyword X=v becomes SetX(v) in the lifecycle)
    #       - Helper function returning VTK: prop = get_default_property()
    #
    # 2. Relationship assignments (_track_relationship_assignments):
    #    a. Property getter: prop = actor.GetProperty()
    #    b. VTK Pythonic property assignments (_track_pythonic_property_assignment):
    #       actor.mapper = my_mapper   (normalized to: actor.SetMapper(my_mapper))
    #       actor.property = my_prop   (normalized to: actor.SetProperty(my_prop))
    #
    # Code Map for visit_Assign:
    #
    #   visit_Assign
    #   │
    #   ├── 1. _handle_call_assignment ─────────────── if node.value is ast.Call
    #   │   │
    #   │   ├── 1.a. _handle_attribute_call_assignment ── func has dot (ast.Attribute)
    #   │   │        └── _track_assignment_targets ────── records var→class mapping
    #   │   │
    #   │   └── 1.b. _handle_direct_call_assignment ───── func has no dot (ast.Name)
    #   │            └── _track_assignment_targets
    #   │                ├── _extract_var_name ────────── gets 'var' or 'self.var'
    #   │                └── _track_pythonic_constructor_keywords ── normalizes X=v → SetX(v)
    #   │                    ├── _extract_keyword_var
    #   │                    └── _track_relationship_from_method ── (shared)
    #   │
    #   └── 2. _track_relationship_assignments ─────── always runs (links objects)
    #       │
    #       ├── 2.a. _track_property_getter_assignment ── prop = actor.GetProperty()
    #       │
    #       └── 2.b. _track_pythonic_property_assignment ── actor.X = v → SetX(v)
    #                └── _track_method_call ───────────── (shared) records + relationships
    #
    # =========================================================================

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements to track VTK class instantiations."""
        self._current_statement = node

        # 1. Handle Call on the right side of the assignment (node.value).
        if isinstance(node.value, ast.Call):
            self._handle_call_assignment(node)

        # 2. Track relationship assignments (GetProperty and mapper attribute assignments)
        self._track_relationship_assignments(node)

        # Continue walking the tree
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # visit_Assign Helpers
    # -------------------------------------------------------------------------

    def _handle_call_assignment(self, node: ast.Assign) -> None:
        """1. Dispatch call assignments to 1.a or 1.b based on func type."""
        # Handle attribute calls: var = vtk.vtkClass() or result = vtkMath.method()
        if isinstance(node.value.func, ast.Attribute):
            self._handle_attribute_call_assignment(node)
        # Direct call: var = vtkClass() or self.var = vtkClass()
        elif isinstance(node.value.func, ast.Name):
            self._handle_direct_call_assignment(node)

    def _handle_attribute_call_assignment(self, node: ast.Assign) -> None:
        """1.a. Attribute calls (have a dot): vtk.vtkClass() or vtkMath.Method()."""
        # Check if the part before the dot is a simple name (not nested attributes)
        if not isinstance(node.value.func.value, ast.Name):
            return

        prefix = node.value.func.value.id

        # Handle vtk module instantiations: var = vtk.vtkClass()
        if prefix == "vtk":
            vtk_class = node.value.func.attr
            # Verify it's a real VTK class
            if self.mcp_client.is_vtk_class(vtk_class):
                self._track_assignment_targets(node, vtk_class)

        # Track static method calls in assignments: result = vtkMath.Distance2BetweenPoints()
        elif prefix.startswith("vtk"):
            vtk_class = prefix
            # Verify it's a real VTK class via MCP
            if self.mcp_client.is_vtk_class(vtk_class):
                self.ctx.static_method_calls[vtk_class].append(node)

    def _handle_direct_call_assignment(self, node: ast.Assign) -> None:
        """1.b. Direct calls (no dot): vtkClass() or helper_function()."""
        func_name = node.value.func.id

        # Only track if it's a VTK class, not a helper function
        if func_name.startswith("vtk") and func_name not in self.helper_methods:
            # Verify it's a real VTK class via MCP
            if self.mcp_client.is_vtk_class(func_name):
                vtk_class = func_name
                # Track assignment and check for mapper= keyword argument
                self._track_assignment_targets(node, vtk_class, check_mapper_keyword=True)

        # Function call that returns a VTK object: var = get_property()
        elif func_name not in self.helper_methods:
            # Check if this function returns a VTK type.
            return_type = self.helper_return_types.get(func_name)
            if return_type and return_type.startswith("vtk"):
                # Verify it's a real VTK class via MCP
                if self.mcp_client.is_vtk_class(return_type):
                    self._track_assignment_targets(node, return_type)

    def _track_relationship_assignments(self, node: ast.Assign) -> None:
        """2. Track relationship assignments (always runs, links objects)."""
        for target in node.targets:
            self._track_property_getter_assignment(target, node)
            self._track_pythonic_property_assignment(target, node)

    def _track_property_getter_assignment(self, target: ast.expr, node: ast.Assign) -> None:
        """2.a. Property getter: prop = actor.GetProperty()."""
        # Must be: simple_var = something.GetProperty()
        if not isinstance(target, ast.Name):
            return
        if not isinstance(node.value, ast.Call):
            return
        if not isinstance(node.value.func, ast.Attribute):
            return

        method_name = node.value.func.attr
        if method_name not in PROPERTY_GETTERS:
            return
        if not isinstance(node.value.func.value, ast.Name):
            return

        # Link property to parent
        parent_var = node.value.func.value.id
        prop_var = target.id
        self.ctx.property_to_parent[prop_var] = parent_var
        self.ctx.parent_to_properties[parent_var].append(prop_var)

    def _track_pythonic_property_assignment(self, target: ast.expr, node: ast.Assign) -> None:
        """2.b. VTK Pythonic property: actor.X = v → normalized to SetX(v)."""
        # Must be: something.attr = value
        if not isinstance(target, ast.Attribute):
            return
        if not isinstance(target.value, ast.Name):
            return

        parent_var = target.value.id
        attr_name = target.attr

        # Only track if parent is a known VTK variable
        if parent_var not in self.ctx.var_to_class:
            return

        # Convert attr to SetX method name (e.g., mapper → SetMapper)
        method_name = f"Set{attr_name.capitalize()}"

        # Extract value as argument string
        try:
            arg_str = ast.unparse(node.value)
        except Exception:
            arg_str = "..."

        # Extract arg variable for relationship tracking
        arg_var = None
        if isinstance(node.value, ast.Name):
            value_var = node.value.id
            if value_var in self.ctx.var_to_class:
                arg_var = value_var

        # Use shared method call tracking
        self._track_method_call(parent_var, method_name, node, args=[arg_str], arg_var=arg_var)

    def _track_assignment_targets(
        self, node: ast.Assign, vtk_class: str, check_mapper_keyword: bool = False
    ) -> None:
        """Record var→class mapping. Called by 1.a and 1.b."""
        for target in node.targets:
            var_name = self._extract_var_name(target)
            if not var_name:
                continue

            # Record variable -> class mapping and statement
            self.ctx.var_to_class[var_name] = vtk_class
            self.ctx.var_statements[var_name].append(node)

            # Check for VTK Pythonic constructor keywords (mapper=, property=)
            if check_mapper_keyword:
                self._track_pythonic_constructor_keywords(node, vtk_class, var_name)

    def _extract_var_name(self, target: ast.expr) -> str | None:
        """Extract 'var' or 'self.var' from assignment target."""
        # Handle var = ...
        if isinstance(target, ast.Name):
            return target.id
        # Handle self.var = ...
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                return f"self.{target.attr}"
        return None

    def _track_pythonic_constructor_keywords(self, node: ast.Assign, vtk_class: str, var_name: str) -> None:
        """1.b helper: Normalize constructor keywords X=v → SetX(v) method calls."""
        for keyword in node.value.keywords:
            if keyword.arg is None:
                continue  # Skip **kwargs

            # Convert keyword to SetX method name (e.g., mapper → SetMapper)
            method_name = f"Set{keyword.arg.capitalize()}"

            # Extract value as argument string
            try:
                arg_str = ast.unparse(keyword.value)
            except Exception:
                arg_str = "..."

            # Extract arg variable for relationship tracking (only for Actor-like classes)
            arg_var = None
            if "Actor" in vtk_class or vtk_class in ACTOR_LIKE_PROPS:
                arg_var = self._extract_keyword_var(keyword)

            # Use shared method call tracking (skip statement since already tracked)
            self.ctx.var_methods[var_name].append(method_name)
            self.ctx.var_method_calls[var_name].append({
                "name": method_name,
                "args": [arg_str]
            })
            if arg_var:
                self._track_relationship_from_method(var_name, method_name, node, arg_var)

    def _extract_keyword_var(self, keyword: ast.keyword) -> str | None:
        """1.b helper: Extract variable from keyword value (mapper=var)."""
        # Try self.attr pattern first
        var_name = self._extract_var_name(keyword.value)
        if var_name:
            return var_name
        # Fall back to simple Name
        if isinstance(keyword.value, ast.Name):
            return keyword.value.id
        return None

    # =========================================================================
    # visit_Expr - Core Visitor Method
    #
    # Tracks VTK method calls (expressions, not assignments).
    #
    # Expression Patterns Handled:
    #
    # 1. Direct method call (_handle_direct_method_call):
    #    - VTK variable: sphere.SetRadius(1.0), actor.SetMapper(mapper)
    #    - Static method: vtkMath.Distance2BetweenPoints(p1, p2)
    #
    # 2. Instance attribute call (_handle_self_attribute_method_call):
    #    - self.mapper.SetInputData(data)
    #    - self.actor.SetProperty(prop)
    #
    # 3. Chained getter call (_handle_chained_method_call):
    #    - actor.GetProperty().SetColor(1, 0, 0)
    #    - renderer.GetActiveCamera().SetPosition(0, 0, 10)
    #
    # Note: VTK Pythonic API (actor.mapper = m) is handled in visit_Assign,
    #       not here, since those are assignment statements.
    #
    # Code Map for visit_Expr:
    #
    #   visit_Expr
    #   │
    #   ├── 1. _handle_direct_method_call ─────────── sphere.SetRadius(1.0)
    #   │       └── _track_method_call ────────────── (shared) records + relationships
    #   │
    #   ├── 2. _handle_self_attribute_method_call ── self.mapper.SetInputData(data)
    #   │       └── _track_method_call
    #   │
    #   └── 3. _handle_chained_method_call ───────── actor.GetProperty().SetColor(1,0,0)
    #
    # =========================================================================

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression statements to track VTK method calls."""
        self._current_statement = node

        # Skip non-method-call expressions
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Attribute):
            self.generic_visit(node)
            return

        # Dispatch to appropriate handler based on call structure
        method_name = node.value.func.attr
        func_value = node.value.func.value

        # 1. Direct: sphere.SetRadius(1.0), vtkMath.Distance2BetweenPoints(p1, p2)
        if isinstance(func_value, ast.Name):
            self._handle_direct_method_call(func_value.id, method_name, node)
        # 2. Instance attribute: self.mapper.SetInputData(data)
        elif isinstance(func_value, ast.Attribute):
            self._handle_self_attribute_method_call(func_value, method_name, node)
        # 3. Chained getter: actor.GetProperty().SetColor(1, 0, 0)
        elif isinstance(func_value, ast.Call):
            self._handle_chained_method_call(func_value, node)

        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # visit_Expr Helpers
    # -------------------------------------------------------------------------

    def _handle_direct_method_call(self, var_name: str, method_name: str, node: ast.Expr) -> None:
        """1. Direct: sphere.SetRadius(1.0), vtkMath.Distance2BetweenPoints(p1, p2)."""
        # Track VTK variable method calls
        if var_name in self.ctx.var_to_class:
            self._track_method_call(var_name, method_name, node)
        # Track static method calls: vtkMath.Distance2BetweenPoints()
        elif var_name.startswith("vtk"):
            if self.mcp_client.is_vtk_class(var_name):
                self.ctx.static_method_calls[var_name].append(node)

    def _handle_self_attribute_method_call(
        self, func_value: ast.Attribute, method_name: str, node: ast.Expr
    ) -> None:
        """2. Instance attribute: self.mapper.SetInputData(data)."""
        if not isinstance(func_value.value, ast.Name):
            return
        if func_value.value.id != "self":
            return

        var_name = f"self.{func_value.attr}"
        if var_name in self.ctx.var_to_class:
            self._track_method_call(var_name, method_name, node)

    def _handle_chained_method_call(self, func_value: ast.Call, node: ast.Expr) -> None:
        """3. Chained getter: actor.GetProperty().SetColor(1, 0, 0)."""
        if not isinstance(func_value.func, ast.Attribute):
            return

        inner_method = func_value.func.attr
        if inner_method not in CHAINABLE_GETTERS:
            return
        if not isinstance(func_value.func.value, ast.Name):
            return

        parent_var = func_value.func.value.id
        if parent_var not in self.ctx.var_to_class:
            return

        # Add statement to parent variable's lifecycle
        self.ctx.var_statements[parent_var].append(node)

        # Track chained property usage for GetProperty* methods
        if inner_method in PROPERTY_GETTERS:
            self.ctx.parent_has_chained_properties.add(parent_var)

    # =========================================================================
    # visit_For - Core Visitor Method
    #
    # Tracks VTK variable usage within for-loops.
    #
    # Decision Logic:
    #
    # 1. Single VTK variable throughout → capture whole loop as one statement
    #    for i in range(8): hexahedron.GetPointIds().SetId(i, i)
    #
    # 2. Multiple VTK variables → recurse into loop body
    #    for actor in actors: actor.SetMapper(mapper); renderer.AddActor(actor)
    #
    # Code Map for visit_For:
    #
    #   visit_For
    #   │
    #   ├── _collect_vtk_vars_per_statement ── analyze VTK vars per statement
    #   │       └── _extract_vtk_var_from_node
    #   │
    #   ├── _is_single_var_loop ──────────── 1. single var? → capture whole loop
    #   │
    #   └── self.visit(stmt) ─────────────── 2. multiple vars? → recurse
    #
    # =========================================================================

    def visit_For(self, node: ast.For) -> None:
        """Visit for-loops to track VTK variable usage within the loop body."""
        self._current_statement = node

        # Analyze which VTK variables each statement uses
        vars_per_stmt = self._collect_vtk_vars_per_statement(node.body)
        all_vars_in_loop = set().union(*vars_per_stmt.values())

        # 1. Single VTK variable → capture whole loop
        if self._is_single_var_loop(vars_per_stmt, all_vars_in_loop):
            single_var = next(iter(all_vars_in_loop))
            self.ctx.var_statements[single_var].append(node)
            return

        # 2. Multiple VTK variables → recurse into loop body
        for stmt in node.body:
            self.visit(stmt)

    # -------------------------------------------------------------------------
    # visit_For Helpers
    # -------------------------------------------------------------------------

    def _collect_vtk_vars_per_statement(self, statements: list[ast.stmt]) -> dict[ast.stmt, set[str]]:
        """Analyze VTK vars per statement."""
        vars_per_stmt: dict[ast.stmt, set[str]] = {}

        for stmt in statements:
            vars_in_stmt: set[str] = set()
            for body_node in ast.walk(stmt):
                var_name = self._extract_vtk_var_from_node(body_node)
                if var_name:
                    vars_in_stmt.add(var_name)
            vars_per_stmt[stmt] = vars_in_stmt

        return vars_per_stmt

    def _extract_vtk_var_from_node(self, node: ast.AST) -> str | None:
        """Extract 'var' or 'self.var' from node if known VTK variable."""
        if not isinstance(node, ast.Attribute):
            return None

        # Check for var.method() pattern
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if var_name in self.ctx.var_to_class:
                return var_name

        # Check for self.var.method() pattern
        elif isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "self":
                var_name = f"self.{node.value.attr}"
                if var_name in self.ctx.var_to_class:
                    return var_name

        return None

    def _is_single_var_loop(self, vars_per_stmt: dict[ast.stmt, set[str]], all_vars: set[str]) -> bool:
        """1. Single var? → capture whole loop."""
        if len(all_vars) != 1:
            return False
        single_var = next(iter(all_vars))
        return all(single_var in vars for vars in vars_per_stmt.values() if vars)

    # -------------------------------------------------------------------------
    # Shared Helper - Method Call Tracking
    #
    # Records method calls like sphere.SetRadius(1.0) or actor.SetMapper(mapper)
    # in the lifecycle context.
    #
    # Called by:
    # - visit_Expr: _handle_direct_method_call, _handle_self_attribute_method_call
    # - visit_Assign: _track_pythonic_property_assignment (actor.mapper = m → SetMapper)
    #
    # Code Map:
    #
    #   _track_method_call
    #   │
    #   ├── _extract_call_arguments ────────── extract args from ast.Call
    #   │
    #   └── _track_relationship_from_method ── SetMapper/SetProperty relationships
    #                                          (also called by _track_pythonic_constructor_keywords)
    #
    # -------------------------------------------------------------------------

    def _track_method_call(
        self,
        var_name: str,
        method_name: str,
        node: ast.stmt,
        args: list[str] | None = None,
        arg_var: str | None = None,
    ) -> None:
        """Record method call in lifecycle context."""
        # Record statement and method
        self.ctx.var_statements[var_name].append(node)
        self.ctx.var_methods[var_name].append(method_name)

        # Extract or use provided arguments
        if args is None and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            args = self._extract_call_arguments(node.value)
        self.ctx.var_method_calls[var_name].append({
            "name": method_name,
            "args": args or []
        })

        # Track special relationship methods
        self._track_relationship_from_method(var_name, method_name, node, arg_var)

    def _extract_call_arguments(self, call_node: ast.Call) -> list[str]:
        """Extract arguments from a call node as strings for synopsis generation.

        Called by: _track_method_call
        """
        arg_strings: list[str] = []
        if hasattr(call_node, 'args'):
            for arg in call_node.args:
                try:
                    arg_strings.append(ast.unparse(arg))
                except Exception:
                    arg_strings.append("...")
        return arg_strings

    def _track_relationship_from_method(
        self,
        var_name: str,
        method_name: str,
        node: ast.stmt,
        arg_var: str | None = None,
    ) -> None:
        """Track SetMapper/SetProperty relationships from method calls.

        Called by:
        - _track_method_call (visit_Expr + Pythonic property assignments)
        - _track_pythonic_constructor_keywords (vtkActor(mapper=m))
        """
        # Extract arg_var from Expr node if not provided
        if arg_var is None and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if node.value.args:
                first_arg = node.value.args[0]
                arg_var = self._extract_var_name(first_arg)
                if not arg_var and isinstance(first_arg, ast.Name):
                    arg_var = first_arg.id

        if not arg_var:
            return

        # Track SetMapper relationship
        if method_name == "SetMapper":
            self.ctx.mapper_to_actor[arg_var] = var_name
            self.ctx.actor_to_mapper[var_name] = arg_var

        # Track SetProperty relationships
        elif method_name in PROPERTY_SETTERS:
            self.ctx.property_to_parent[arg_var] = var_name
            self.ctx.parent_to_properties[var_name].append(arg_var)

    # -------------------------------------------------------------------------
    # Class Helper - Helper Return Type Extraction
    #
    # Called by: __init__()
    # -------------------------------------------------------------------------

    def _extract_helper_return_types(self) -> dict[str, str]:
        """Extract VTK return types from helper functions by analyzing their AST.

        Handles two patterns:
        1. Direct instantiation: return vtkClass()
        2. Variable return: return var (where var = vtkClass() earlier)

        Returns:
            Mapping of helper function names to their VTK return types.
        """
        helper_return_types: dict[str, str] = {}

        # 1. Iterate over each helper function name (e.g., "make_hexahedron")
        for helper_name in self.helper_methods:
            # Skip if helper not in parsed function definitions
            if helper_name not in self.function_defs:
                continue
            # Get the AST node for this helper function
            func_node = self.function_defs[helper_name]

            # 2. Walk all nodes in the function to find return statements
            for node in ast.walk(func_node):
                # Skip non-return nodes and returns without a value (bare return)
                if not (isinstance(node, ast.Return) and node.value):
                    continue

                # node.value is what's being returned (e.g., vtkClass() or var)
                return_value = node.value
                vtk_class: str | None = None

                # 3a. Pattern 1: return vtkClass() — direct instantiation
                #     AST: Return(value=Call(func=Name(id='vtkHexahedron')))
                if isinstance(return_value, ast.Call) and isinstance(return_value.func, ast.Name):
                    # return_value.func.id is the function/class name being called
                    class_name = return_value.func.id
                    # Verify it's a VTK class (starts with "vtk" and exists in VTK)
                    if class_name.startswith("vtk") and self.mcp_client.is_vtk_class(class_name):
                        vtk_class = class_name

                # 3b. Pattern 2: return var — variable holding VTK object
                #     AST: Return(value=Name(id='hexahedron'))
                #     Need to find: hexahedron = vtkHexahedron() earlier in function
                elif isinstance(return_value, ast.Name):
                    # return_value.id is the variable name being returned
                    var_name = return_value.id
                    # Walk function again to find assignment to this variable
                    for stmt in ast.walk(func_node):
                        # Look for assignment statements (var = ...)
                        if isinstance(stmt, ast.Assign):
                            # Check each target (handles a = b = vtkClass())
                            for target in stmt.targets:
                                # target.id is the variable being assigned to
                                if isinstance(target, ast.Name) and target.id == var_name:
                                    # stmt.value is the RHS; check if it's a VTK call
                                    if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                                        class_name = stmt.value.func.id
                                        if class_name.startswith("vtk") and self.mcp_client.is_vtk_class(class_name):
                                            vtk_class = class_name
                                            break  # Found the assignment, stop searching

                # 4. If we found a VTK return type, record it and move to next helper
                if vtk_class:
                    helper_return_types[helper_name] = vtk_class
                    break  # Only need first return with VTK type

        return helper_return_types
