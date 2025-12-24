"""Build semantic chunks from grouped VTK object lifecycles."""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from vtk_rag.mcp import VTKClient

from .models import CodeChunk
from ..query import SemanticQuery


class SemanticChunk:
    """Builds a CodeChunk from grouped VTK object lifecycles."""

    def __init__(self, code: str, example_id: str, filename: str, mcp_client: VTKClient) -> None:
        self.code = code
        self.example_id = example_id
        self.filename = filename
        self.mcp_client = mcp_client
        self.chunk_counter = 0
        self.semantic_query = SemanticQuery(mcp_client)

    def _build_chunk(self, group: list[dict], helper_function: str | None = None) -> dict:
        """Build a chunk from one or more related lifecycles.

        Args:
            group: List of lifecycle dictionaries to include in this chunk.
            helper_function: Name of the helper function (e.g., 'make_hexahedron') for query generation.

        Returns:
            CodeChunk as a dictionary.
        """
        # Collect all statements, classes, and variables from all lifecycles
        all_statements = []
        all_classes = []
        all_variables = []
        code_lines: list[str] = []
        role = ""

        for lifecycle in group:
            all_classes.append(lifecycle["class"])
            if lifecycle.get("variable"):
                all_variables.append(lifecycle["variable"])
            # Add property classes (e.g., vtkProperty from actor.GetProperty())
            for prop in lifecycle.get("properties", []):
                if prop["class"] and prop["class"] not in all_classes:
                    all_classes.append(prop["class"])
            all_statements.extend(lifecycle["statements"])
            if not role:
                role = lifecycle["type"]

        # Build code from lifecycle statements
        for stmt in all_statements:
            stmt_source = ast.get_source_segment(self.code, stmt)
            if stmt_source:
                code_lines.append(stmt_source)

        lifecycle_code = "\n".join(code_lines)

        # Rewrite vtk.vtkClass() → vtkClass() and self.var → var
        rewritten_code = re.sub(r'\bvtk\.(vtk\w+)', r'\1', lifecycle_code)
        rewritten_code = re.sub(r'\bself\.(\w+)', r'\1', rewritten_code)

        # Deduplicate classes maintaining order
        vtk_classes = list(dict.fromkeys(all_classes))

        # Generate imports
        imports = self._imports(vtk_classes)

        # Imports + Code chunk
        full_chunk_code = "\n".join(imports) + "\n\n" + rewritten_code

        # Determine datatypes based on role
        input_datatype = ""
        output_datatype = ""
        if role == "renderer":
            input_datatype = "vtkActor"
            output_datatype = "vtkRenderer"
        elif role in ("infrastructure", "scene"):
            input_datatype = "vtkRenderer"
        elif role == "properties":
            # Find mapper to get input datatype
            mapper_class = next((c for c in all_classes if "Mapper" in c), None)
            actor_class = next((c for c in all_classes if "Actor" in c), None)
            if mapper_class:
                input_datatype = self.mcp_client.get_class_input_datatype(mapper_class) or ""
                output_datatype = "vtkActor"
            elif actor_class:
                input_datatype = "vtkMapper"
                output_datatype = "vtkActor"
        else:
            if vtk_classes:
                input_datatype = self.mcp_client.get_class_input_datatype(vtk_classes[0]) or ""
                output_datatype = self.mcp_client.get_class_output_datatype(vtk_classes[0]) or ""

        # Compute visibility score
        if not vtk_classes:
            visibility_score = 0.5
        else:
            scores = []
            for vtk_class in vtk_classes:
                score = self.mcp_client.get_class_visibility(vtk_class)
                scores.append(score if score is not None else 0.5)
            visibility_score = sum(scores) / len(scores)

        # Collect and deduplicate methods from lifecycles
        class_methods: dict[str, list[str]] = {}
        class_method_calls: dict[str, list[dict]] = {}
        for lifecycle in group:
            cls = lifecycle["class"]
            # Dedupe methods by name
            class_methods[cls] = list(dict.fromkeys(
                class_methods.get(cls, []) + lifecycle.get("methods", [])
            ))
            # Dedupe method_calls by name (keep first occurrence)
            existing = {mc["name"] for mc in class_method_calls.get(cls, [])}
            class_method_calls[cls] = class_method_calls.get(cls, []) + [
                mc for mc in lifecycle.get("method_calls", []) if mc["name"] not in existing
            ]

        # Build queries from classes and methods
        semantic_queries = self.semantic_query.build(vtk_classes, class_method_calls)

        # Add helper function query if this is a helper function
        if helper_function:
            semantic_queries.append(self.semantic_query.function_name_to_query(helper_function))

        # Build action_phrase from classes and methods
        action_phrase = self._action_phrase(vtk_classes)

        # Build synopsis from classes and methods
        synopsis = self._synopsis(vtk_classes, class_method_calls)

        # Build vtk_classes list
        vtk_classes_data = [
            {
                "class": cls,
                "variables": [lc["variable"] for lc in group if lc["class"] == cls],
                "methods": class_methods.get(cls, []),
            }
            for cls in vtk_classes
        ]

        # Build CodeChunk
        self.chunk_counter += 1
        chunk = CodeChunk(
            chunk_id=f"{self.filename}_chunk_{self.chunk_counter}",
            example_id=self.example_id,
            action_phrase=action_phrase,
            synopsis=synopsis,
            role=role,
            visibility_score=visibility_score,
            input_datatype=input_datatype,
            output_datatype=output_datatype,
            content=full_chunk_code,
            variable_name=", ".join(all_variables) if all_variables else "",
            vtk_classes=vtk_classes_data,
            queries=semantic_queries,
        )

        return chunk.to_dict()

    def _action_phrase(self, vtk_classes: list[str]) -> str:
        """Build action phrase from VTK classes.

        Args:
            vtk_classes: List of VTK class names.

        Returns:
            Combined action phrase (e.g., "Sphere creation → Actor setup").
        """
        action_phrases = []

        for vtk_class in vtk_classes:
            action_phrase = self.mcp_client.get_class_action_phrase(vtk_class)
            if action_phrase:
                action_phrase = action_phrase[0].upper() + action_phrase[1:]
            else:
                # Fallback: vtkEventDataDevice -> "Event data device"
                name = vtk_class[3:] if vtk_class.startswith("vtk") else vtk_class
                action_phrase = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).capitalize()
            action_phrases.append(action_phrase)

        return " → ".join(action_phrases) if action_phrases else ""

    def _synopsis(
        self, vtk_classes: list[str], class_method_calls: dict[str, list[dict]]
    ) -> str:
        """Build synopsis from VTK classes and their method calls.

        Args:
            vtk_classes: List of VTK class names.
            class_method_calls: Dict mapping class names to method call dicts.

        Returns:
            Synopsis string describing class usage with method details.
        """
        synopses = []

        for vtk_class in vtk_classes:
            # Get action phrase for this class
            action_phrase = self.mcp_client.get_class_action_phrase(vtk_class)
            if action_phrase:
                action_phrase = action_phrase[0].upper() + action_phrase[1:]
            else:
                name = vtk_class[3:] if vtk_class.startswith("vtk") else vtk_class
                action_phrase = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).capitalize()

            # Get method calls for this class
            method_calls = class_method_calls.get(vtk_class, [])[:8]

            # Parse method calls into phrases
            method_phrases = [
                phrase for method_call in method_calls
                if (phrase := self._method_phrase(method_call["name"], method_call.get("args", [])))
            ]

            # Build synopsis part
            if method_phrases:
                synopses.append(f"{action_phrase} with {', '.join(method_phrases)}")
            else:
                synopses.append(action_phrase)

        return " → ".join(synopses) if synopses else ""

    def _method_phrase(self, method_name: str, args: list[str]) -> str | None:
        """Convert a VTK method call to a natural language phrase.

        Examples:
            SetRadius(2.5) -> "radius set to 2.5"
            SetCenter(1.0, 2.0, 3.0) -> "center set to (1.0, 2.0, 3.0)"
            AddRenderer(ren) -> "renderer added"
            RemoveActor(actor) -> "actor removed"
            CreateDefaultLookupTable() -> "default lookup table created"
        """
        # CamelCase to lowercase words: ThetaResolution -> "theta resolution"
        def to_words(name: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()

        # Skip getters and lifecycle methods
        if method_name.startswith("Get"):
            return None
        # Skip lifecycle/execution methods - they don't describe configuration
        lifecycle_methods = {
            "Update", "Initialize", "Render", "Start", "Modified", "Delete",
            "ShallowCopy", "DeepCopy", "Execute", "Finalize", "Allocate", "Release",
        }
        if method_name in lifecycle_methods:
            return None

        # Prefix patterns: Set*, Add*, Remove*, Create*, Clear*, Reset*
        prefixes = {
            "Set": ("set to", True),      # (verb, needs_args_format)
            "Add": ("added", False),
            "Remove": ("removed", False),
            "Create": ("created", False),
            "Clear": ("cleared", False),
            "Reset": ("reset", False),
            "Build": ("built", False),
            "Compute": ("computed", False),
        }
        for prefix, (verb, needs_args) in prefixes.items():
            if method_name.startswith(prefix):
                thing = to_words(method_name[len(prefix):])
                if needs_args and args:
                    if len(args) == 1:
                        return f"{thing} {verb} {args[0]}"
                    return f"{thing} {verb} ({', '.join(args)})"
                return f"{thing} {verb}" if thing else f"{verb}"

        # Suffix patterns: *On, *Off (toggle methods)
        suffixes = {
            "On": "enabled",
            "Off": "disabled",
        }
        for suffix, verb in suffixes.items():
            if method_name.endswith(suffix):
                return f"{to_words(method_name[:-len(suffix)])} {verb}"

        # Other methods with args
        if args:
            return f"{to_words(method_name)} with {', '.join(args)}"
        return None

    def _imports(self, vtk_classes: list[str]) -> list[str]:
        """Generate import statements for VTK classes.

        Args:
            vtk_classes: List of VTK class names.

        Returns:
            List of import statements.
        """
        imports = []

        # Side-effect imports (VTK backend registration)
        needs_opengl2 = any("RenderWindow" in c and "Interactor" not in c for c in vtk_classes)
        needs_interaction = any("Interactor" in c for c in vtk_classes)

        if needs_opengl2:
            imports.append("import vtkmodules.vtkRenderingOpenGL2")
        if needs_interaction:
            imports.append("import vtkmodules.vtkInteractionStyle")

        # Class imports grouped by module
        module_to_classes: dict[str, list[str]] = defaultdict(list)
        for cls, module in self.mcp_client.get_class_modules(set(vtk_classes)).items():
            module_to_classes[module].append(cls)

        for module in sorted(module_to_classes):
            classes = ", ".join(sorted(set(module_to_classes[module])))
            imports.append(f"from {module} import {classes}")

        return imports
