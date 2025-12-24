"""Analyze VTK object lifecycles and group them into semantic categories."""

from __future__ import annotations

import ast
from collections import defaultdict

from vtk_rag.mcp import VTKClient

from ..vtk_categories import (
    ACTOR_LIKE_PROPS,
    CHAINABLE_GETTERS,
    PROPERTY_GETTERS,
    PROPERTY_MAPPINGS,
    PROPERTY_SETTERS,
)
from .models import MethodCall, VTKLifecycle


class LifecycleAnalyzer:
    """Analyzes VTK lifecycles and groups them."""

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
        # Extract helper return types from AST
        self.helper_return_types = self._extract_helper_return_types()

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
        # Track variable assignments to VTK classes.
        var_to_class: dict[str, str] = initial_var_to_class.copy() if initial_var_to_class else {}
        # Track all statements involving each VTK variable.
        var_statements: dict[str, list[ast.stmt]] = defaultdict(list)
        # Track method calls per variable (var -> [method_names]).
        var_methods: dict[str, list[str]] = defaultdict(list)
        # Track method calls with arguments (var -> [MethodCall]).
        var_method_calls: dict[str, list[MethodCall]] = defaultdict(list)
        # Track property relationships (property_var -> parent_actor_var).
        property_to_parent: dict[str, str] = {}
        # Track parent to explicit properties (actor_var -> [property_vars]).
        parent_to_properties: dict[str, list[str]] = defaultdict(list)
        # Track parents with chained property usage (actor_var -> True).
        # e.g., actor.GetProperty().SetColor() vs prop = actor.GetProperty(); prop.SetColor()
        parent_has_chained_properties: set[str] = set()
        # Track mapper relationships (mapper_var -> actor_var).
        mapper_to_actor: dict[str, str] = {}
        # Track actor to mapper (actor_var -> mapper_var).
        actor_to_mapper: dict[str, str] = {}
        # Track static method calls (vtk_class -> [statements]).
        static_method_calls: dict[str, list[ast.stmt]] = defaultdict(list)

        class LifecycleVisitor(ast.NodeVisitor):
            def __init__(self, helper_methods: set[str], helper_return_types: dict[str, str], mcp_client: VTKClient) -> None:
                self.helper_methods = helper_methods
                self.helper_return_types = helper_return_types
                self.mcp_client = mcp_client
                self._current_statement = None

            def visit_Assign(self, node: ast.Assign) -> None:
                """Visit assignment statements to track VTK class instantiations.

                Handles patterns like:
                    var = vtkClass()
                    var = vtk.vtkClass()
                    self.var = vtkClass()
                    result = vtkMath.Distance2BetweenPoints()
                """
                self._current_statement = node

                # Handle Call on the right side of the assignment (node.value).
                if isinstance(node.value, ast.Call):

                    # Handle attribute calls: var = vtk.vtkClass() or result = vtkMath.method()
                    if isinstance(node.value.func, ast.Attribute):
                        # Check if the part before the dot is a simple name (not nested attributes)
                        if isinstance(node.value.func.value, ast.Name):
                            # Handle vtk module instantiations: var = vtk.vtkClass()
                            if node.value.func.value.id == "vtk":
                                vtk_class = node.value.func.attr
                                # Verify it's a real VTK class
                                if self.mcp_client.is_vtk_class(vtk_class):
                                    self._track_assignment_targets(node, vtk_class)
                            # Track static method calls in assignments: result = vtkMath.Distance2BetweenPoints()
                            elif node.value.func.value.id.startswith("vtk"):
                                vtk_class = node.value.func.value.id
                                # Verify it's a real VTK class via MCP
                                if self.mcp_client.is_vtk_class(vtk_class):
                                    static_method_calls[vtk_class].append(node)
                    # Direct call: var = vtkClass() or self.var = vtkClass()
                    elif isinstance(node.value.func, ast.Name):
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

                # Track relationship assignments (GetProperty and mapper attribute assignments)
                self._track_relationship_assignments(node)

                # Continue walking the tree
                self.generic_visit(node)

            def _track_relationship_assignments(self, node: ast.Assign) -> None:
                """Helper to track relationship assignments in targets.

                Handles two patterns:
                1. prop = actor.GetProperty() - Links property variable to parent
                2. actor.mapper = my_mapper - Links mapper to actor via attribute assignment
                """
                for target in node.targets:
                    # Handle prop = actor.GetProperty()
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Attribute):
                            method_name = node.value.func.attr
                            if method_name in PROPERTY_GETTERS:
                                if isinstance(node.value.func.value, ast.Name):
                                    parent_var = node.value.func.value.id
                                    prop_var = target.id
                                    property_to_parent[prop_var] = parent_var
                                    parent_to_properties[parent_var].append(prop_var)

                    # Handle actor.mapper = my_mapper
                    elif isinstance(target, ast.Attribute) and isinstance(node.value, ast.Name):
                        if target.attr == "mapper" and isinstance(target.value, ast.Name):
                            actor_var = target.value.id
                            mapper_var = node.value.id
                            if actor_var in var_to_class and mapper_var in var_to_class:
                                mapper_to_actor[mapper_var] = actor_var
                                actor_to_mapper[actor_var] = mapper_var

            def _track_assignment_targets(self, node: ast.Assign, vtk_class: str, check_mapper_keyword: bool = False) -> None:
                """Helper to track VTK class assignments to variables.

                Handles both regular variables (var = vtkClass()) and self.attributes (self.var = vtkClass()).
                Optionally checks for mapper= keyword argument in Actor-like constructors.
                """
                # Handle targets like var = vtkClass() or self.var = vtkClass() and multiple targets like var1 = var2 = vtkClass()
                for target in node.targets:
                    var_name = None

                    # Handle var = vtkClass()
                    if isinstance(target, ast.Name):
                        var_name = target.id
                    # Handle self.var = vtkClass()
                    elif isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == "self":
                            var_name = f"self.{target.attr}"

                    # Track the variable and the statement
                    if var_name:
                        var_to_class[var_name] = vtk_class
                        var_statements[var_name].append(node)

                        # Check for mapper= keyword argument if requested
                        if check_mapper_keyword and ("Actor" in vtk_class or vtk_class in ACTOR_LIKE_PROPS):
                            for keyword in node.value.keywords:
                                if keyword.arg == "mapper":
                                    mapper_var = None
                                    # Handle self.mapper
                                    if isinstance(keyword.value, ast.Attribute):
                                        if isinstance(keyword.value.value, ast.Name) and keyword.value.value.id == "self":
                                            mapper_var = f"self.{keyword.value.attr}"
                                    # Handle regular mapper variable
                                    elif isinstance(keyword.value, ast.Name):
                                        mapper_var = keyword.value.id

                                    # track bidirectional mapper to actor mapping
                                    if mapper_var:
                                        mapper_to_actor[mapper_var] = var_name
                                        actor_to_mapper[var_name] = mapper_var

            # Expression Statements: *.method()
            def visit_Expr(self, node: ast.Expr) -> None:
                """Visit expression statements to track VTK method calls.

                Handles three patterns of method calls:
                1. var.method() - Direct variable or static class method calls
                2. self.var.method() - Class instance variable method calls
                3. var.getter().method() - Chained method calls through getters

                Tracks SetProperty, SetMapper, and inline property usage.
                """
                self._current_statement = node

                # Not a method call
                if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Attribute):
                    # Continue walking the tree
                    self.generic_visit(node)
                    return

                # Get method name and value
                method_name = node.value.func.attr
                func_value = node.value.func.value

                # Handle direct method calls: var.method() or vtkClass.method()
                if isinstance(func_value, ast.Name):
                    var_name = func_value.id

                    # Track VTK variable method calls
                    if var_name in var_to_class:
                        self._track_method_call(var_name, method_name, node)

                    # Track static method calls: vtkMath.Distance2BetweenPoints()
                    elif var_name.startswith("vtk"):
                        if self.mcp_client.is_vtk_class(var_name):
                            static_method_calls[var_name].append(node)

                # Handle self.attribute method calls: self.mapper.SetInputData()
                elif isinstance(func_value, ast.Attribute):
                    if isinstance(func_value.value, ast.Name) and func_value.value.id == "self":
                        var_name = f"self.{func_value.attr}"
                        if var_name in var_to_class:
                            self._track_method_call(var_name, method_name, node)

                # Handle chained calls: actor.GetProperty().SetColor(...), renderer.GetActiveCamera().SetPosition(...)
                elif isinstance(func_value, ast.Call) and isinstance(func_value.func, ast.Attribute):
                    inner_method = func_value.func.attr

                    if inner_method in CHAINABLE_GETTERS and isinstance(func_value.func.value, ast.Name):
                        parent_var = func_value.func.value.id
                        if parent_var in var_to_class:
                            # Add statement to parent variable's lifecycle
                            var_statements[parent_var].append(node)
                            # Track chained property usage for GetProperty* methods
                            if inner_method in PROPERTY_GETTERS:
                                parent_has_chained_properties.add(parent_var)

                # Continue walking the tree
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                """Visit for-loops to track VTK variable usage within the loop body.

                Captures patterns like:
                    for i in range(n):
                        hexahedron.GetPointIds().SetId(i, i)

                Only adds the entire for-loop if ALL statements in the loop body use the same
                VTK variable (e.g., a loop that only sets point IDs). Otherwise, we recurse
                into the loop body to capture individual statements.
                """
                self._current_statement = node

                # Find which VTK variables are used in each statement of the loop body
                vars_per_stmt: dict[ast.stmt, set[str]] = {}
                all_vars_in_loop: set[str] = set()

                for stmt in node.body:
                    vars_in_stmt: set[str] = set()
                    for body_node in ast.walk(stmt):
                        # Check for var.method() calls
                        if isinstance(body_node, ast.Attribute) and isinstance(body_node.value, ast.Name):
                            var_name = body_node.value.id
                            if var_name in var_to_class:
                                vars_in_stmt.add(var_name)
                        # Check for self.var.method() calls
                        elif isinstance(body_node, ast.Attribute) and isinstance(body_node.value, ast.Attribute):
                            if isinstance(body_node.value.value, ast.Name) and body_node.value.value.id == "self":
                                var_name = f"self.{body_node.value.attr}"
                                if var_name in var_to_class:
                                    vars_in_stmt.add(var_name)
                    vars_per_stmt[stmt] = vars_in_stmt
                    all_vars_in_loop.update(vars_in_stmt)

                # If all statements use the same single VTK variable, capture the whole loop
                # (e.g., a loop that only does hexahedron.GetPointIds().SetId(i, i))
                if len(all_vars_in_loop) == 1:
                    single_var = next(iter(all_vars_in_loop))
                    if all(single_var in vars for vars in vars_per_stmt.values() if vars):
                        var_statements[single_var].append(node)
                        return

                # Otherwise, recurse into the loop body to capture individual statements
                for stmt in node.body:
                    self.visit(stmt)

            def _track_method_call(self, var_name: str, method_name: str, node: ast.Expr) -> None:
                """Helper to track VTK variable method calls and relationship tracking.

                Always adds the statement to the variable's lifecycle.

                Additionally tracks two special relationships for proper properties grouping:
                1. SetProperty calls - Links property objects to their parent (actors, volumes,
                   text actors, etc.) so we can group them together in the same chunk.
                2. SetMapper calls - Links mappers to actors so we can create complete
                   properties chunks (mapper → actor).
                """
                # Add statement to variable's lifecycle
                var_statements[var_name].append(node)
                # Track method name for this variable
                var_methods[var_name].append(method_name)

                # Extract arguments as strings for synopsis generation
                call_node = node.value
                arg_strings = []
                if hasattr(call_node, 'args'):
                    for arg in call_node.args:
                        try:
                            arg_strings.append(ast.unparse(arg))
                        except Exception:
                            arg_strings.append("...")
                # Track method call with arguments
                var_method_calls[var_name].append({
                    "name": method_name,
                    "args": arg_strings
                })

                # Track SetProperty calls
                if method_name in PROPERTY_SETTERS:
                    # Handle self.property = value - actor.SetProperty(self.prop)
                    if node.value.args and isinstance(node.value.args[0], ast.Attribute):
                        if isinstance(node.value.args[0].value, ast.Name) and node.value.args[0].value.id == "self":
                            prop_var = f"self.{node.value.args[0].attr}"
                            property_to_parent[prop_var] = var_name
                            parent_to_properties[var_name].append(prop_var)
                    # Handle property = value - actor.SetProperty(prop)
                    elif node.value.args and isinstance(node.value.args[0], ast.Name):
                        prop_var = node.value.args[0].id
                        property_to_parent[prop_var] = var_name
                        parent_to_properties[var_name].append(prop_var)

                # Track SetMapper calls
                elif method_name == "SetMapper" and node.value.args:
                    mapper_arg = node.value.args[0]
                    # Handle mapper = value - actor.SetMapper(mapper)
                    if isinstance(mapper_arg, ast.Name):
                        mapper_var = mapper_arg.id
                        mapper_to_actor[mapper_var] = var_name
                        actor_to_mapper[var_name] = mapper_var
                    # Handle self.mapper = value - self.actor.SetMapper(self.mapper)
                    elif isinstance(mapper_arg, ast.Attribute):
                        if isinstance(mapper_arg.value, ast.Name) and mapper_arg.value.id == "self":
                            mapper_var = f"self.{mapper_arg.attr}"
                            mapper_to_actor[mapper_var] = var_name
                            actor_to_mapper[var_name] = mapper_var

        LifecycleVisitor(self.helper_methods, self.helper_return_types, self.mcp_client).visit(func_node)

        # Build lifecycle objects for each VTK variable.
        lifecycles = []
        processed_vars = set()

        for var_name, vtk_class in var_to_class.items():
            if var_name in processed_vars:
                continue

            # Check if this is a property that belongs to a parent.
            if var_name in property_to_parent:
                # Skip properties; they'll be included with their parent.
                continue

            # Collect all statements for this variable and its properties.
            all_statements = list(var_statements[var_name])
            related_properties = []

            # Include explicit property statements if this variable has them.
            if var_name in parent_to_properties:
                for prop_var in parent_to_properties[var_name]:
                    if prop_var in var_to_class:
                        # Explicit property variable (assigned to a variable).
                        related_properties.append({
                            "variable": prop_var,
                            "class": var_to_class.get(prop_var),
                        })
                        all_statements.extend(var_statements[prop_var])

            # Sort all statements by their line number to preserve execution order.
            all_statements.sort(key=lambda stmt: stmt.lineno)

            # Add chained property marker if chained usage detected.
            if var_name in parent_has_chained_properties:
                # Infer property class from parent actor/volume class
                prop_class = None
                for prop_cls, parent_classes in PROPERTY_MAPPINGS.items():
                    if vtk_class in parent_classes:
                        prop_class = prop_cls
                        break
                if prop_class:
                    related_properties.append({
                        "variable": "inline",
                        "class": prop_class,
                    })

            # Get the VTK class role from MCP
            vtk_type = self.mcp_client.get_class_role(vtk_class) or "utility"

            # Get mapper/actor relationships from local tracking
            mapper_var = actor_to_mapper.get(var_name)
            actor_var = mapper_to_actor.get(var_name)

            # Deduplicate methods while preserving order
            methods = var_methods.get(var_name, [])
            seen = set()
            unique_methods = [m for m in methods if not (m in seen or seen.add(m))]

            # Deduplicate method_calls by name while preserving order and keeping first occurrence
            method_calls = var_method_calls.get(var_name, [])
            seen_calls = set()
            unique_method_calls = []
            for mc in method_calls:
                if mc["name"] not in seen_calls:
                    seen_calls.add(mc["name"])
                    unique_method_calls.append(mc)

            lifecycles.append({
                "variable": var_name,
                "class": vtk_class,
                "type": vtk_type,
                "statements": all_statements,
                "properties": related_properties,
                "mapper": mapper_var,  # If this is an actor, which mapper does it use?
                "actor": actor_var,   # If this is a mapper, which actor uses it?
                "methods": unique_methods,  # Method names called on this object (legacy)
                "method_calls": unique_method_calls,  # Method calls with arguments
            })

            processed_vars.add(var_name)
            # Mark properties as processed.
            for prop in related_properties:
                processed_vars.add(prop["variable"])

        # Add static method call chunks (one chunk per class, combining all calls)
        for vtk_class, statements in static_method_calls.items():
            if statements:
                # Get the VTK class role from MCP
                vtk_type = self.mcp_client.get_class_role(vtk_class) or "utility"
                # Sort statements by line number to preserve execution order
                sorted_statements = sorted(statements, key=lambda stmt: stmt.lineno)

                # Extract method names and arguments from static calls
                static_methods = []
                static_method_calls_list = []
                for stmt in sorted_statements:
                    call_node = None
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        call_node = stmt.value
                    elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        call_node = stmt.value

                    if call_node and isinstance(call_node.func, ast.Attribute):
                        method_name = call_node.func.attr
                        static_methods.append(method_name)
                        # Extract arguments
                        arg_strings = []
                        for arg in call_node.args:
                            try:
                                arg_strings.append(ast.unparse(arg))
                            except Exception:
                                arg_strings.append("...")
                        static_method_calls_list.append({
                            "name": method_name,
                            "args": arg_strings
                        })

                # Deduplicate while preserving order
                seen = set()
                unique_static_methods = [m for m in static_methods if not (m in seen or seen.add(m))]

                # Deduplicate method_calls by name
                seen_calls = set()
                unique_static_method_calls = []
                for mc in static_method_calls_list:
                    if mc["name"] not in seen_calls:
                        seen_calls.add(mc["name"])
                        unique_static_method_calls.append(mc)

                lifecycles.append({
                    "variable": None,  # No variable for static calls
                    "class": vtk_class,
                    "type": vtk_type,  # Use module name
                    "statements": sorted_statements,  # All calls to this class combined
                    "properties": [],
                    "mapper": None,
                    "actor": None,
                    "methods": unique_static_methods,  # Static method names (legacy)
                    "method_calls": unique_static_method_calls,  # Method calls with arguments
                })

        return lifecycles

    def _group_lifecycles(self, lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
        """Group lifecycles into chunks based on their role.

        - properties: grouped by mapper/actor relationships
        - infrastructure: grouped together (window + interactor), sorted so window comes first
        - all other roles: each lifecycle is its own chunk
        """
        properties_lifecycles: list[VTKLifecycle] = []
        infrastructure_lifecycles: list[VTKLifecycle] = []
        single_lifecycle_groups: list[list[VTKLifecycle]] = []

        for lc in lifecycles:
            role = lc["type"]
            if role == "properties":
                properties_lifecycles.append(lc)
            elif role == "infrastructure":
                infrastructure_lifecycles.append(lc)
            else:
                # All other roles: input, filter, output, renderer, scene, utility, color
                single_lifecycle_groups.append([lc])

        # Group properties by their relationships (mapper/actor/property)
        properties_groups = self._group_properties(properties_lifecycles)

        # Group infrastructure together with proper ordering
        infrastructure_group = self._group_infrastructure(infrastructure_lifecycles)

        return infrastructure_group + properties_groups + single_lifecycle_groups

    def _group_infrastructure(self, infrastructure_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
        """Group infrastructure lifecycles with proper ordering.

        Order:
        1. RenderWindow first
        2. Interactor (without Start() statement)
        3. InteractorStyles and other infrastructure
        4. Start() statement last

        The interactor lifecycle is split: all statements except Start() come early,
        then styles, then Start() at the very end.
        """
        if not infrastructure_lifecycles:
            return []

        window_lifecycles: list[VTKLifecycle] = []
        interactor_lifecycle: VTKLifecycle | None = None
        style_lifecycles: list[VTKLifecycle] = []

        for lc in infrastructure_lifecycles:
            vtk_class = lc["class"]
            if "RenderWindow" in vtk_class and "Interactor" not in vtk_class:
                window_lifecycles.append(lc)
            elif "Interactor" in vtk_class and "Style" not in vtk_class:
                interactor_lifecycle = lc
            else:
                style_lifecycles.append(lc)

        result: list[VTKLifecycle] = []
        result.extend(window_lifecycles)

        if interactor_lifecycle:
            # Split interactor: statements before Start(), then styles, then Start()
            start_stmt = None
            other_stmts = []
            for stmt in interactor_lifecycle.get("statements", []):
                # Check if this statement is a Start() call
                if self._is_start_call(stmt):
                    start_stmt = stmt
                else:
                    other_stmts.append(stmt)

            # Create interactor lifecycle without Start()
            interactor_without_start = dict(interactor_lifecycle)
            interactor_without_start["statements"] = other_stmts
            # Remove Start from methods list
            methods = [m for m in interactor_lifecycle.get("methods", []) if m != "Start"]
            interactor_without_start["methods"] = methods
            result.append(interactor_without_start)

            # Add styles
            result.extend(style_lifecycles)

            # Add Start() as a minimal lifecycle at the end
            if start_stmt:
                start_lifecycle: VTKLifecycle = {
                    "variable": interactor_lifecycle["variable"],
                    "class": interactor_lifecycle["class"],
                    "type": interactor_lifecycle["type"],
                    "statements": [start_stmt],
                    "properties": [],
                    "mapper": None,
                    "actor": None,
                    "methods": ["Start"],
                    "method_calls": [],
                }
                result.append(start_lifecycle)
        else:
            result.extend(style_lifecycles)

        return [result] if result else []

    def _is_start_call(self, stmt: ast.stmt) -> bool:
        """Check if a statement is an interactor.Start() call."""
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "Start":
                return True
        return False

    def _group_properties(self, properties_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
        """Group properties elements (mappers, actors) with their related properties."""
        groups = []
        processed = set()

        # Create a lookup for quick access
        var_to_lifecycle = {lc["variable"]: lc for lc in properties_lifecycles if lc["variable"]}

        # Start with actors (they're at the top of the pipeline)
        for lc in properties_lifecycles:
            if lc["variable"] in processed:
                continue

            # Only start groups from actors or standalone mappers
            # Check class name since type now contains role
            is_actor = "Actor" in lc["class"] and not lc["class"].endswith("Property")
            is_actor = is_actor or lc["class"] in ACTOR_LIKE_PROPS
            if is_actor:
                group = []

                # If this actor has a mapper, add the mapper first
                mapper_var = lc.get("mapper")
                if mapper_var and mapper_var in var_to_lifecycle:
                    mapper_lc = var_to_lifecycle[mapper_var]

                    # Add mapper's properties first
                    for prop in mapper_lc.get("properties", []):
                        prop_var = prop["variable"]
                        if prop_var != "inline" and prop_var in var_to_lifecycle:
                            group.append(var_to_lifecycle[prop_var])
                            processed.add(prop_var)

                    # Then add the mapper
                    group.append(mapper_lc)
                    processed.add(mapper_var)

                # Add actor's properties
                for prop in lc.get("properties", []):
                    prop_var = prop["variable"]
                    if prop_var != "inline" and prop_var in var_to_lifecycle and prop_var not in processed:
                        group.append(var_to_lifecycle[prop_var])
                        processed.add(prop_var)

                # Finally add the actor
                group.append(lc)
                processed.add(lc["variable"])

                groups.append(group)

        # Add any remaining mappers that don't have actors
        for lc in properties_lifecycles:
            if lc["variable"] not in processed and "Mapper" in lc["class"]:
                group = []

                # Add mapper's properties first
                for prop in lc.get("properties", []):
                    prop_var = prop["variable"]
                    if prop_var != "inline" and prop_var in var_to_lifecycle and prop_var not in processed:
                        group.append(var_to_lifecycle[prop_var])
                        processed.add(prop_var)

                # Then add the mapper
                group.append(lc)
                processed.add(lc["variable"])

                groups.append(group)

        # Add any remaining properties classes (LUTs, transfer functions, textures, etc.)
        # that weren't grouped with actors or mappers
        for lc in properties_lifecycles:
            if lc["variable"] not in processed:
                groups.append([lc])
                processed.add(lc["variable"])

        return groups

    def _merge_class_lifecycles(self, lifecycles: list[VTKLifecycle]) -> list[VTKLifecycle]:
        """Merge lifecycles for the same variable (e.g., self.mapper) across methods.

        When analyzing class methods, the same self.variable may appear in multiple methods.
        This merges them into a single lifecycle with combined statements and properties.
        """
        var_to_lifecycle: dict[str, VTKLifecycle] = {}

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
                    "methods": lc.get("methods", [])[:],
                    "method_calls": lc.get("method_calls", [])[:],
                }
            else:
                # Merge statements from this lifecycle
                existing = var_to_lifecycle[base_var]
                existing["statements"].extend(lc["statements"])
                existing["properties"].extend(lc["properties"])
                existing["methods"].extend(lc.get("methods", []))
                existing["method_calls"].extend(lc.get("method_calls", []))
                # Update mapper/actor if not set
                if not existing.get("mapper") and lc.get("mapper"):
                    existing["mapper"] = lc["mapper"]
                if not existing.get("actor") and lc.get("actor"):
                    existing["actor"] = lc["actor"]

        return list(var_to_lifecycle.values())

    def _build_class_var_map(self, class_node: ast.ClassDef) -> dict[str, str]:
        """Build a map of self.variable -> VTK class for all variables in a class.

        Walks all methods to find self.variable = vtkClass() assignments.
        Used as initial context when analyzing class methods.
        """
        class_var_to_class: dict[str, str] = {}

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
                                if self.mcp_client.is_vtk_class(vtk_class):
                                    for target in node.targets:
                                        if isinstance(target, ast.Attribute):
                                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                                var_name = f"self.{target.attr}"
                                                class_var_to_class[var_name] = vtk_class

        return class_var_to_class

    def _extract_helper_return_types(self) -> dict[str, str]:
        """Extract VTK return types from helper functions by analyzing their AST.

        Handles two patterns:
        1. Direct instantiation: return vtkClass()
        2. Variable return: return var (where var = vtkClass() earlier)

        Returns:
            Mapping of helper function names to their VTK return types.
        """
        helper_return_types: dict[str, str] = {}

        for helper_name in self.helper_methods:
            if helper_name not in self.function_defs:
                continue
            func_node = self.function_defs[helper_name]

            # Walk the function AST to find return statements
            for node in ast.walk(func_node):
                if not (isinstance(node, ast.Return) and node.value):
                    continue

                return_value = node.value
                vtk_class: str | None = None

                # Pattern 1: return vtkClass() - direct instantiation
                if isinstance(return_value, ast.Call) and isinstance(return_value.func, ast.Name):
                    class_name = return_value.func.id
                    if class_name.startswith("vtk") and self.mcp_client.is_vtk_class(class_name):
                        vtk_class = class_name

                # Pattern 2: return var (where var = vtkClass() earlier)
                elif isinstance(return_value, ast.Name):
                    var_name = return_value.id
                    for stmt in ast.walk(func_node):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == var_name:
                                    if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                                        class_name = stmt.value.func.id
                                        if class_name.startswith("vtk") and self.mcp_client.is_vtk_class(class_name):
                                            vtk_class = class_name
                                            break

                if vtk_class:
                    helper_return_types[helper_name] = vtk_class
                    break

        return helper_return_types
