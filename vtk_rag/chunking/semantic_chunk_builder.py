"""Build semantic chunks from grouped VTK object lifecycles."""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from .code_chunk import CodeChunk
from .code_query_generator import CodeQueryGenerator
from .vtk_categories import (
    INPUT_DATATYPE_METHODS,
    OUTPUT_DATATYPE_METHODS,
    SELF_CONTAINED_ACTORS,
)
from vtk_rag.mcp import VTK_API_CLIENT

# Visibility weights for computing chunk visibility score
VISIBILITY_WEIGHTS = {
    "very_likely": 1.0,
    "likely": 0.8,
    "maybe": 0.5,
    "unlikely": 0.2,
    "internal_only": 0.0,
    "deprecated": 0.0,
}


class SemanticChunkBuilder:
    """Build semantic chunks from grouped VTK object lifecycles."""

    # Shared query generator instance
    _query_generator: CodeQueryGenerator | None = None

    @classmethod
    def get_query_generator(cls) -> CodeQueryGenerator:
        """Get or create the shared query generator."""
        if cls._query_generator is None:
            cls._query_generator = CodeQueryGenerator()
        return cls._query_generator

    def __init__(self, code: str, example_id: str, filename: str, class_defs: dict[str, ast.ClassDef]):
        self.code = code
        self.example_id = example_id
        self.filename = filename
        self.class_defs = class_defs
        self.chunk_counter = 0
        self.query_generator = self.get_query_generator()

    def _build_chunk(self, func_name: str, group: list[dict], is_class_chunk: bool = False, class_name: str | None = None) -> dict:
        """Build a chunk from one or more related lifecycles.

        Handles both single lifecycles and grouped lifecycles (visualization pipelines,
        rendering infrastructure, etc.) in a unified way.

        Args:
            func_name: Name of the function being chunked
            group: List of lifecycle dictionaries to include in this chunk
            is_class_chunk: If True, strip self. from code and add class_context metadata
            class_name: Name of the class (only used if is_class_chunk=True)
        """
        # Determine if this is a single or grouped chunk
        is_single = len(group) == 1

        # Check if this is a single lifecycle with a self-contained actor
        if is_single:
            lifecycle = group[0]
            if lifecycle["type"] == "Actors" and lifecycle["class"] in SELF_CONTAINED_ACTORS:
                # Override type to Visualization Pipeline
                lifecycle = lifecycle.copy()
                lifecycle["type"] = "Visualization Pipeline"
                group = [lifecycle]

        # Collect all statements, classes, and properties from all lifecycles
        all_statements = []
        all_classes = []
        all_properties = []
        group_type = None
        var_name = None  # For single lifecycles

        for lifecycle in group:
            all_classes.append(lifecycle["class"])
            all_properties.extend(lifecycle.get("properties", []))
            all_statements.extend(lifecycle["statements"])
            if group_type is None:
                group_type = lifecycle["type"]
            if is_single:
                var_name = lifecycle["variable"]

        # For single lifecycles, check if it's a user-defined class
        is_user_defined_class = False
        if is_single:
            vtk_class = all_classes[0]
            is_user_defined_class = vtk_class in self.class_defs

        # Extract source code
        code_lines = []

        # If it's a user-defined class, include the class definition first
        if is_user_defined_class:
            class_node = self.class_defs[vtk_class]
            class_source = ast.get_source_segment(self.code, class_node)
            if class_source:
                code_lines.append(class_source)
                code_lines.append("")  # Add blank line after class definition

        for stmt in all_statements:
            stmt_source = ast.get_source_segment(self.code, stmt)
            if stmt_source:
                # Strip self. for class chunks to make code reusable
                if is_class_chunk:
                    stmt_source = re.sub(r'\bself\.(\w+)', r'\1', stmt_source)
                code_lines.append(stmt_source)

        lifecycle_code = "\n".join(code_lines)
        rewritten_code = self._rewrite_vtk_calls(lifecycle_code)

        # Collect all VTK classes used
        used_classes = list(set(all_classes))
        for prop in all_properties:
            if prop["class"] not in used_classes:
                used_classes.append(prop["class"])

        # Generate imports
        class_imports = self._get_class_imports(used_classes)

        # Add module imports for all classes in group
        module_imports = []
        for cls in all_classes:
            module_imports.extend(self._get_required_module_imports(cls))
        # Remove duplicates while preserving order
        seen = set()
        module_imports = [x for x in module_imports if not (x in seen or seen.add(x))]

        # Combine imports + code
        all_imports = module_imports + class_imports
        full_chunk_code = "\n".join(all_imports) + "\n\n" + rewritten_code

        # Generate title, description, and determine datatypes
        title, description, group_type, input_datatype, output_datatype = self._generate_chunk_metadata(
            group, group_type, all_classes, is_single, var_name
        )

        # Collect methods from lifecycles (already extracted during lifecycle analysis)
        class_methods = defaultdict(list)
        class_method_calls = defaultdict(list)  # With arguments
        for lifecycle in group:
            vtk_class = lifecycle["class"]
            methods = lifecycle.get("methods", [])
            method_calls = lifecycle.get("method_calls", [])
            class_methods[vtk_class].extend(methods)
            class_method_calls[vtk_class].extend(method_calls)

        # Deduplicate methods per class while preserving order
        for cls in class_methods:
            seen = set()
            class_methods[cls] = [m for m in class_methods[cls] if not (m in seen or seen.add(m))]

        # Deduplicate method_calls per class by name while preserving order
        for cls in class_method_calls:
            seen = set()
            unique_calls = []
            for mc in class_method_calls[cls]:
                if mc["name"] not in seen:
                    seen.add(mc["name"])
                    unique_calls.append(mc)
            class_method_calls[cls] = unique_calls

        # Extract new metadata: roles, visibility, synopsis
        roles, visibility_score, synopsis = self._extract_semantic_metadata(
            used_classes, class_method_calls
        )

        # Build metadata
        chunk_metadata = {
            "vtk_classes": [
                {
                    "class": cls,
                    "variables": [lc["variable"] for lc in group if lc["class"] == cls],
                    "methods": class_methods.get(cls, []),
                }
                for cls in used_classes
            ],
            "properties": all_properties,
            "class_imports": class_imports,
            "module_imports": module_imports,
            "input_datatype": input_datatype,
            "output_datatype": output_datatype,
            "roles": roles,
            "visibility_score": visibility_score,
        }

        # Add class_context for class chunks
        if is_class_chunk and class_name:
            chunk_metadata["class_context"] = class_name

        # Generate chunk ID using filename + counter
        self.chunk_counter += 1
        chunk_id = f"{self.filename}_chunk_{self.chunk_counter}"

        # Build CodeChunk with consistent fields for all chunk types
        chunk = CodeChunk(
            chunk_id=chunk_id,
            example_id=self.example_id,
            type=group_type,
            function_name=func_name,
            title=title,
            description=description,
            synopsis=synopsis,
            roles=roles,
            visibility_score=visibility_score,
            input_datatype=input_datatype or "N/A",
            output_datatype=output_datatype or "N/A",
            content=full_chunk_code,
            metadata=chunk_metadata,
            variable_name=var_name if is_single else "N/A",
            vtk_class=all_classes[0] if is_single else "N/A",
        )

        # Generate queries for this chunk
        chunk.queries = self.query_generator.generate_queries(chunk.to_dict())

        return chunk.to_dict()

    def _generate_chunk_metadata(self, group: list[dict], group_type: str, all_classes: list[str],
                                   is_single: bool, var_name: str | None) -> tuple[str, str, str, str | None, str | None]:
        """Generate title, description, type, and datatypes for a chunk.

        Returns: (title, description, group_type, input_datatype, output_datatype)
        """
        if is_single:
            # Single lifecycle
            vtk_class = all_classes[0]
            properties = group[0]["properties"]

            # Extract datatypes via MCP
            datatypes = self._extract_datatypes(vtk_class)
            input_datatype = datatypes["input_datatype"]
            output_datatype = datatypes["output_datatype"]

            # Generate title and description
            if var_name is None:
                # Static method calls (no variable)
                title = f"{vtk_class} Utility Methods"
                description = f"Utility method calls on {vtk_class}"
            else:
                title = f"{vtk_class} Lifecycle"
                description = f"Lifecycle for {var_name} ({vtk_class})"
                if properties:
                    prop_names = ", ".join(p["class"] for p in properties)
                    description += f" with properties: {prop_names}"
        else:
            # Grouped lifecycles
            if group_type in ["Renderers", "Render windows & interactors", "Rendering Infrastructure"]:
                title = "Rendering Infrastructure"
                var_names = [lc["variable"] for lc in group]
                description = f"Rendering infrastructure: {', '.join(var_names)}"
                group_type = "Rendering Infrastructure"  # Override type
                input_datatype = "vtkActor"
                output_datatype = "N/A"
            else:
                # Visualization Pipeline
                var_names = [lc["variable"] for lc in group]
                title = f"Visualization Pipeline: {' → '.join(var_names)}"
                description = f"Visualization pipeline with {', '.join(all_classes)}"
                group_type = "Visualization Pipeline"  # Override type

                # Find the mapper to get input datatype
                mapper_class = None
                for cls in all_classes:
                    if "Mapper" in cls:
                        mapper_class = cls
                        break

                if mapper_class:
                    mapper_datatypes = self._extract_datatypes(mapper_class)
                    input_datatype = mapper_datatypes["input_datatype"]
                else:
                    input_datatype = "N/A"
                output_datatype = "vtkActor"

        return title, description, group_type, input_datatype, output_datatype

    def _extract_datatypes(self, vtk_class: str) -> dict[str, str | None]:
        """Extract input and output datatypes using MCP vtk_get_method_info.

        VTK pipeline patterns:
        - Output: GetOutputPort() for connections, GetOutput() for data objects
        - Input: SetInputConnection() for pipelines, SetInputData() for static data

        Note: Classes that don't produce datasets (mappers, actors, renderers, writers)
        won't have GetOutput() methods.
        """
        input_datatype = None
        output_datatype = None

        # Try multiple output methods to find output datatype
        for method_name in OUTPUT_DATATYPE_METHODS:
            if output_datatype:  # Already found
                break
            method_info = VTK_API_CLIENT.get_method_info(vtk_class, method_name)
            if method_info:
                # Format: "GetOutput(self) -> vtkPolyData"
                content_str = method_info.get("content", "")
                match = re.search(r'->\s*(vtk\w+)', content_str)
                if match:
                    output_datatype = match.group(1)

        # Try multiple input methods to find input datatype
        for method_name in INPUT_DATATYPE_METHODS:
            if input_datatype:  # Already found
                break
            method_info = VTK_API_CLIENT.get_method_info(vtk_class, method_name)
            if method_info:
                content_str = method_info.get("content", "")

                if method_name == "SetInputConnection":
                    # SetInputConnection takes vtkAlgorithmOutput, need to infer from context
                    # For now, we'll try to get it from SetInputData if available
                    continue
                else:
                    # Format: "SetInputData(self, __a:vtkPolyData)"
                    match = re.search(r':\s*(vtk\w+)', content_str)
                    if match:
                        input_datatype = match.group(1)

        return {
            "input_datatype": input_datatype,
            "output_datatype": output_datatype
        }

    def _extract_semantic_metadata(
        self, vtk_classes: list[str], class_method_calls: dict[str, list[dict]]
    ) -> tuple[list[str], float, str]:
        """Extract semantic metadata for RAG indexing: roles, visibility, synopsis.

        Args:
            vtk_classes: List of VTK classes used in the chunk
            class_method_calls: Dict mapping class names to method calls with arguments
                               Each call is {"name": str, "args": List[str]}

        Returns:
            (roles, visibility_score, synopsis) tuple where:
            - roles: List of unique functional roles (e.g., ['source_geometric', 'mapper_polydata'])
            - visibility_score: Weighted average visibility (0.0-1.0)
            - synopsis: Natural language description for query matching
        """
        roles = []
        visibility_values = []
        synopsis_parts = []

        for vtk_class in vtk_classes:
            # Skip user-defined classes (not in VTK API)
            if vtk_class in self.class_defs:
                continue

            # Get role
            role = VTK_API_CLIENT.get_class_role(vtk_class)
            if role and role not in roles:
                roles.append(role)

            # Get visibility
            visibility_str = VTK_API_CLIENT.get_class_visibility(vtk_class)
            if visibility_str:
                weight = VISIBILITY_WEIGHTS.get(visibility_str, 0.5)
                visibility_values.append(weight)
            else:
                visibility_values.append(0.5)  # Default to maybe

            # Get action phrase for concise synopsis building
            # e.g., "polygonal sphere creation" instead of full synopsis
            action_phrase = VTK_API_CLIENT.get_class_action_phrase(vtk_class)
            if action_phrase:
                # Capitalize first letter for sentence start
                action_phrase = action_phrase[0].upper() + action_phrase[1:]

            # Fall back to class name if no action phrase
            if not action_phrase:
                action_phrase = vtk_class

            # Parse method calls into natural language phrases
            # e.g., SetThetaResolution(32) -> "theta resolution set to 32"
            method_calls = class_method_calls.get(vtk_class, [])
            method_phrases = []
            for method_call in method_calls[:8]:  # Limit to first 8 methods
                method_name = method_call["name"]
                args = method_call.get("args", [])
                phrase = self._method_call_to_phrase(method_name, args)
                if phrase:
                    method_phrases.append(phrase)

            # Combine action phrase with method phrases
            if action_phrase and method_phrases:
                synopsis_parts.append(f"{action_phrase} with {', '.join(method_phrases)}")
            elif action_phrase:
                synopsis_parts.append(action_phrase)

        # Compute weighted visibility score (average of all classes)
        if visibility_values:
            visibility_score = sum(visibility_values) / len(visibility_values)
        else:
            visibility_score = 0.5  # Default

        # Build synopsis string
        synopsis = " | ".join(synopsis_parts) if synopsis_parts else ""

        return roles, round(visibility_score, 2), synopsis

    def _method_call_to_phrase(self, method_name: str, args: list[str]) -> str | None:
        """Convert a VTK method call to a natural language phrase.

        Examples:
            SetRadius(2.5) -> "radius set to 2.5"
            SetThetaResolution(32) -> "theta resolution set to 32"
            SetCenter(1.0, 2.0, 3.0) -> "center set to (1.0, 2.0, 3.0)"
            GetOutput() -> None (getters don't configure anything)
            AddRenderer(ren) -> "renderer added"
        """
        # Skip getters - they don't configure anything
        if method_name.startswith("Get"):
            return None

        # Handle Set* methods - the most common case
        if method_name.startswith("Set"):
            # Extract property name: SetThetaResolution -> ThetaResolution
            prop_name = method_name[3:]
            # Convert CamelCase to lowercase with spaces: ThetaResolution -> theta resolution
            readable_name = self._camel_to_words(prop_name)

            if args:
                # Format args: single value or tuple
                if len(args) == 1:
                    return f"{readable_name} set to {args[0]}"
                else:
                    return f"{readable_name} set to ({', '.join(args)})"
            else:
                return f"{readable_name} set"

        # Handle Add* methods
        if method_name.startswith("Add"):
            thing = method_name[3:]
            readable_name = self._camel_to_words(thing)
            return f"{readable_name} added"

        # Handle Enable*/Disable* methods
        if method_name.endswith("On"):
            prop_name = method_name[:-2]
            readable_name = self._camel_to_words(prop_name)
            return f"{readable_name} enabled"

        if method_name.endswith("Off"):
            prop_name = method_name[:-3]
            readable_name = self._camel_to_words(prop_name)
            return f"{readable_name} disabled"

        # Handle Update/Initialize/Render etc. - action verbs
        if method_name in ("Update", "Initialize", "Render", "Start", "Modified"):
            return None  # These are lifecycle methods, not configuration

        # For other methods, just convert the name
        readable_name = self._camel_to_words(method_name)
        if args:
            return f"{readable_name} with {', '.join(args)}"
        return None  # Skip methods without args that we don't recognize

    def _camel_to_words(self, name: str) -> str:
        """Convert CamelCase to lowercase words.

        Examples:
            ThetaResolution -> theta resolution
            SetInputConnection -> set input connection
            RGB -> rgb
        """
        # Insert space before uppercase letters (but not at start)
        result = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        return result.lower()

    def _rewrite_vtk_calls(self, source: str) -> str:
        """Rewrite vtk.vtkClass() → vtkClass() in the source code."""
        try:
            # Parse the source into an AST.
            tree = ast.parse(source)

            # Walk the AST and rewrite vtk.vtkClass attribute accesses.
            class VtkRewriter(ast.NodeTransformer):
                def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                    # Check if this is vtk.vtkSomething
                    if (
                        isinstance(node.value, ast.Name)
                        and node.value.id == "vtk"
                        and node.attr.startswith("vtk")
                    ):
                        # Replace vtk.vtkClass with just vtkClass (as a Name node).
                        return ast.Name(id=node.attr, ctx=node.ctx)
                    return self.generic_visit(node)

            rewritten_tree = VtkRewriter().visit(tree)
            ast.fix_missing_locations(rewritten_tree)

            # Convert the rewritten AST back to source code.
            return ast.unparse(rewritten_tree)
        except SyntaxError:
            # If parsing fails, return original source.
            return source

    def _get_required_module_imports(self, vtk_class: str) -> list[str]:
        """Get required module imports based on VTK class type.

        VTK requires additional module imports beyond the classes used in
        the lifecycle.
        """
        module_imports = []

        # RenderWindow needs OpenGL2 backend
        if "RenderWindow" in vtk_class and "Interactor" not in vtk_class:
            module_imports.append("import vtkmodules.vtkRenderingOpenGL2")

        # RenderWindowInteractor needs interaction style
        if "Interactor" in vtk_class:
            module_imports.append("import vtkmodules.vtkInteractionStyle")

        return module_imports

    def _get_class_imports(self, vtk_classes: list[str]) -> list[str]:
        """Generate import statements for VTK classes using MCP.

        Returns valid Python import statements like:
        ["from vtkmodules.vtkRenderingCore import vtkActor, vtkProperty", ...]

        Used both for generating actual imports in code and for metadata.
        """
        import_lines = []
        # Group imports by module.
        module_to_classes: dict[str, list[str]] = defaultdict(list)

        # Use VTK_API_CLIENT (MCP) to get authoritative module information
        resolved = VTK_API_CLIENT.resolve(set(vtk_classes))

        for vtk_class in vtk_classes:
            if vtk_class in resolved:
                module_path = resolved[vtk_class]
                # module_path is like "vtkmodules.vtkRenderingCore"
                module_to_classes[module_path].append(vtk_class)

        # Generate one import line per module.
        for module in sorted(module_to_classes.keys()):
            classes = sorted(set(module_to_classes[module]))
            import_lines.append(f"from {module} import {', '.join(classes)}")

        return import_lines
