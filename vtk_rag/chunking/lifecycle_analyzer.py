"""Analyze VTK object lifecycles and group them into semantic categories."""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import TypedDict

from .vtk_categories import (
    ACTOR_LIKE_PROPS,
    CHAINABLE_GETTERS,
    IMAGE_MAPPERS,
    PROPERTY_GETTERS,
    PROPERTY_MAPPINGS,
    PROPERTY_SETTERS,
    SELF_CONTAINED_ACTORS,
)
from .vtk_class_resolver import VTK_CLASS_RESOLVER


class MethodCall(TypedDict):
    """Structure of a method call with its arguments."""
    name: str  # Method name (e.g., "SetRadius")
    args: list[str]  # String representations of arguments (e.g., ["10", "True"])


class VTKLifecycle(TypedDict, total=False):
    """Structure of a VTK lifecycle dictionary."""
    variable: str | None  # Variable name or None for static methods
    class_: str  # VTK class name (using class_ to avoid keyword)
    type: str  # Classification (e.g., "Mappers", "vtkmodules.vtkRenderingCore")
    statements: list[ast.stmt]  # AST statements for this lifecycle
    properties: list[dict[str, str]]  # Related properties with 'variable' and 'class' keys
    mapper: str | None  # Mapper variable if this is an actor
    actor: str | None  # Actor variable if this is a mapper
    methods: list[str]  # Method names called on this VTK object (legacy, for compatibility)
    method_calls: list[MethodCall]  # Method calls with arguments


class LifecycleAnalyzer:
    """Analyzes VTK lifecycles and groups them."""

    def __init__(self, code: str, helper_return_types: dict[str, str],
                 helper_methods: set[str], class_defs: dict[str, ast.ClassDef]):
        """Initialize the lifecycle analyzer.

        Args:
            code: Source code being analyzed.
            helper_return_types: Mapping of helper function names to their VTK return types.
            helper_methods: Set of helper function names to exclude from lifecycle tracking.
            class_defs: Mapping of class names to their AST definitions.
        """
        self.code = code
        self.helper_return_types = helper_return_types
        self.helper_methods = helper_methods
        self.class_defs = class_defs

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
                - type: VTK type classification (e.g., "Mappers", "Actors")
                - statements: AST nodes for all statements involving this variable
                - properties: Related property objects
                - mapper/actor: Relationship tracking for visualization pipelines
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
            def __init__(self, helper_methods: set[str], helper_return_types: dict[str, str]) -> None:
                self.helper_methods = helper_methods
                self.helper_return_types = helper_return_types
                self._current_statement = None

            # Assignment Statements: var = value
            def visit_Assign(self, node: ast.Assign) -> None:
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
                                # Verify it's a real VTK class via MCP
                                resolved = VTK_CLASS_RESOLVER.resolve({vtk_class})
                                if vtk_class in resolved:
                                    self._track_assignment_targets(node, vtk_class)
                            # Track static method calls in assignments: result = vtkMath.Distance2BetweenPoints()
                            elif node.value.func.value.id.startswith("vtk"):
                                vtk_class = node.value.func.value.id
                                # Verify it's a real VTK class via MCP
                                resolved = VTK_CLASS_RESOLVER.resolve({vtk_class})
                                if vtk_class in resolved:
                                    static_method_calls[vtk_class].append(node)
                    # Direct call: var = vtkClass() or self.var = vtkClass()
                    elif isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        # Only track if it's a VTK class, not a helper function
                        if func_name.startswith("vtk") and func_name not in self.helper_methods:
                            # Verify it's a real VTK class via MCP
                            resolved = VTK_CLASS_RESOLVER.resolve({func_name})
                            if func_name in resolved:
                                vtk_class = func_name
                                # Track assignment and check for mapper= keyword argument
                                self._track_assignment_targets(node, vtk_class, check_mapper_keyword=True)
                        # Function call that returns a VTK object: var = get_property()
                        elif func_name not in self.helper_methods:
                            # Check if this function returns a VTK type.
                            return_type = self.helper_return_types.get(func_name)
                            if return_type and return_type.startswith("vtk"):
                                # Verify it's a real VTK class via MCP
                                resolved = VTK_CLASS_RESOLVER.resolve({return_type})
                                if return_type in resolved:
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
                        resolved = VTK_CLASS_RESOLVER.resolve({var_name})
                        if var_name in resolved:
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

            def _track_method_call(self, var_name: str, method_name: str, node: ast.Expr) -> None:
                """Helper to track VTK variable method calls and relationship tracking.

                Always adds the statement to the variable's lifecycle.

                Additionally tracks two special relationships for proper visualization pipeline grouping:
                1. SetProperty calls - Links property objects to their parent (actors, volumes,
                   text actors, etc.) so we can group them together in the same chunk.
                2. SetMapper calls - Links mappers to actors so we can create complete
                   visualization pipeline chunks (mapper → actor).
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

        LifecycleVisitor(self.helper_methods, self.helper_return_types).visit(func_node)

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

            # Classify the VTK class type.
            vtk_type = self._classify_vtk_class(vtk_class)

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
                # Classify by MCP to get module name (e.g., vtkmodules.vtkCommonCore for vtkMath)
                vtk_type = self._classify_vtk_class(vtk_class)
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
        """Group lifecycles into chunks based on their relationships and types."""
        # Categorize lifecycles
        query_elements = []  # Sources, readers, filters - keep separate
        visualization_tuples = []  # (property, mapper, actor) groups
        rendering_infrastructure = []  # Renderers, windows, interactors - combine all
        other = []

        for lc in lifecycles:
            vtk_type = lc["type"]
            vtk_class = lc["class"]

            # Query elements - keep separate (Sources, Readers, Writers, Filters)
            # These are classified by MCP, so check class name patterns
            if ("Source" in vtk_class or "Reader" in vtk_class or "Writer" in vtk_class or
                "Filter" in vtk_class or vtk_type == "Data objects / datasets"):
                query_elements.append([lc])  # Each is its own group

            # Rendering infrastructure - combine all into one group
            elif vtk_type in ["Renderers", "Render windows & interactors", "Rendering Infrastructure"]:
                rendering_infrastructure.append(lc)

            # Visualization elements - will be grouped by relationships
            elif vtk_type in ["Mappers", "Actors"]:
                visualization_tuples.append(lc)

            # Everything else - keep separate (module-based types, etc.)
            else:
                other.append([lc])

        # Group visualization pipelines by their relationships
        vis_groups = self._group_visualization_pipelines(visualization_tuples)

        # Group rendering infrastructure
        rendering_groups = self._group_rendering_infrastructure(rendering_infrastructure)

        # Combine all groups: rendering infrastructure, visualization pipelines, query elements, and other
        return rendering_groups + vis_groups + query_elements + other

    def _group_rendering_infrastructure(self, rendering_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
        """Group rendering infrastructure elements (renderers, windows, interactors, cameras, lights).

        Combines all rendering infrastructure into one group, sorted by dependency order:
        cameras/lights → renderer → render window → interactor
        """
        if not rendering_lifecycles:
            return []

        # Sort by rendering order
        def rendering_order(lc: VTKLifecycle) -> int:
            vtk_class = lc["class"]
            # Cameras and lights first (added to renderer)
            if vtk_class in ["vtkCamera", "vtkLight", "vtkLightKit", "vtkLightActor", "vtkCameraActor"]:
                return 1
            # Render passes (added to renderer)
            elif vtk_class.endswith("Pass") or "Pass" in vtk_class:
                return 1
            # Renderer second (cameras/lights added to it, then added to window)
            elif "Renderer" in vtk_class and "Window" not in vtk_class:
                return 2
            # Render window third (renderer added to it, then set on interactor)
            elif "RenderWindow" in vtk_class and "Interactor" not in vtk_class:
                return 3
            # Interactor last (window set on it, then Start())
            elif "Interactor" in vtk_class:
                return 4
            return 5

        rendering_lifecycles.sort(key=rendering_order)

        # Combine all into one group
        return [rendering_lifecycles]

    def _group_visualization_pipelines(self, vis_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
        """Group visualization pipeline elements (mappers, actors) with their properties."""
        groups = []
        processed = set()

        # Create a lookup for quick access
        var_to_lifecycle = {lc["variable"]: lc for lc in vis_lifecycles if lc["variable"]}

        # Start with actors (they're at the top of the pipeline)
        for lc in vis_lifecycles:
            if lc["variable"] in processed:
                continue

            # Only start groups from actors or standalone mappers
            if lc["type"] == "Actors":
                group = []
                vtk_class = lc["class"]

                # Check if this is a self-contained actor
                is_self_contained = vtk_class in SELF_CONTAINED_ACTORS

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
                elif is_self_contained:
                    # For self-contained actors that support explicit mapper assignment
                    # vtkImageActor.SetMapper(vtkImageMapper3D*) and vtkTextActor.SetMapper(vtkTextMapper*)
                    if vtk_class == 'vtkImageActor':
                        # Look for image mappers to group with vtkImageActor
                        for mapper_lc in vis_lifecycles:
                            if mapper_lc["variable"] not in processed and mapper_lc["type"] == "Mappers":
                                if mapper_lc["class"] in IMAGE_MAPPERS:
                                    group.append(mapper_lc)
                                    processed.add(mapper_lc["variable"])
                                    break
                    elif vtk_class == 'vtkTextActor':
                        # Look for vtkTextMapper to group with vtkTextActor
                        for mapper_lc in vis_lifecycles:
                            if mapper_lc["variable"] not in processed and mapper_lc["type"] == "Mappers":
                                if mapper_lc["class"] == 'vtkTextMapper':
                                    group.append(mapper_lc)
                                    processed.add(mapper_lc["variable"])
                                    break

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
        for lc in vis_lifecycles:
            if lc["variable"] not in processed and lc["type"] == "Mappers":
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

        return groups

    def _classify_vtk_class(self, vtk_class: str) -> str:
        """Classify a VTK class - pattern-based for visualization pipelines/rendering infrastructure,
        MCP for everything else."""

        # Critical for rendering infrastructure (check FIRST before Actor pattern)
        if vtk_class in ["vtkCamera", "vtkLight", "vtkLightKit", "vtkLightActor", "vtkCameraActor"]:
            return "Rendering Infrastructure"
        elif "Renderer" in vtk_class and "RenderWindow" not in vtk_class:
            return "Renderers"
        elif "RenderWindow" in vtk_class or "Interactor" in vtk_class:
            return "Render windows & interactors"
        elif vtk_class.endswith("Pass") or "Pass" in vtk_class:
            return "Rendering Infrastructure"

        # Critical for visualization pipelines (check AFTER rendering infrastructure)
        elif "Mapper" in vtk_class:
            return "Mappers"
        elif "Actor" in vtk_class and not vtk_class.endswith("Property"):
            return "Actors"
        elif vtk_class in ACTOR_LIKE_PROPS:
            return "Actors"

        # Everything else - use MCP to get module name
        resolved = VTK_CLASS_RESOLVER.resolve({vtk_class})
        if vtk_class in resolved:
            # Return the full module path like "vtkmodules.vtkRenderingCore"
            return resolved[vtk_class]

        # Safety net: if MCP doesn't know about this class, return "Other"
        # Note: In practice, this never happens - MCP has complete VTK coverage
        return "Other"
