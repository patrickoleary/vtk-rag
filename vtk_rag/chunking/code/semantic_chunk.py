"""Build semantic chunks from grouped VTK object lifecycles.

This module transforms lifecycle groups into CodeChunk objects suitable for
embedding and retrieval. Each chunk contains code, metadata, and semantic
queries for similarity search.

Called by:
    CodeChunker._extract_function_chunks(), _extract_module_chunks(),
    _extract_class_chunks() in chunker.py

Code Map:
    SemanticChunk
        build_chunk()                    # main orchestrator (public API)
            ├── _collect_from_lifecycles()   # gather statements, classes, variables
            ├── _build_code()                # AST → code string with rewrites
            ├── _determine_datatypes()       # input/output datatypes by role
            ├── _compute_visibility()        # average visibility score
            ├── _collect_methods()           # merge and dedupe methods per class
            ├── _action_phrase()             # human-readable action description
            ├── _synopsis()                  # detailed description with methods
            │       └── _method_phrase()     # method call → natural language
            └── _imports()                   # generate import statements
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from vtk_rag.mcp import VTKClient

from ..query import SemanticQuery
from .lifecycle.models import LifecycleData
from .lifecycle.utils import dedupe_method_calls, dedupe_preserve_order
from .models import CodeChunk


class SemanticChunk:
    """Builds a CodeChunk from grouped VTK object lifecycles."""

    def __init__(self, code: str, example_id: str, filename: str, mcp_client: VTKClient) -> None:
        self.code = code
        self.example_id = example_id
        self.filename = filename
        self.mcp_client = mcp_client
        self.chunk_counter = 0
        self.semantic_query = SemanticQuery(mcp_client)

    def build_chunk(self, group: list[dict], helper_function: str | None = None) -> dict:
        """Build a CodeChunk from one or more related lifecycles.

        Output: CodeChunk dictionary with code, metadata, and semantic queries.
        """
        # 1. Collect data from lifecycles
        data = self._collect_from_lifecycles(group)
        vtk_classes = dedupe_preserve_order(data.classes)

        # 2. Build code content
        full_code = self._build_code(data.statements, vtk_classes)

        # 3. Determine datatypes based on role
        input_dt, output_dt = self._determine_datatypes(data.role, vtk_classes)

        # 4. Compute visibility score
        visibility = self._compute_visibility(vtk_classes)

        # 5. Collect methods per class
        class_methods, class_method_calls = self._collect_methods(group)

        # 6. Build semantic queries
        queries = self.semantic_query.build(vtk_classes, class_method_calls)
        if helper_function:
            queries.append(self.semantic_query.function_name_to_query(helper_function))

        # 7. Build descriptions
        action_phrase = self._action_phrase(vtk_classes)
        synopsis = self._synopsis(vtk_classes, class_method_calls)

        # 8. Assemble CodeChunk
        self.chunk_counter += 1
        chunk = CodeChunk(
            chunk_id=f"{self.filename}_chunk_{self.chunk_counter}",
            example_id=self.example_id,
            action_phrase=action_phrase,
            synopsis=synopsis,
            role=data.role,
            visibility_score=visibility,
            input_datatype=input_dt,
            output_datatype=output_dt,
            content=full_code,
            variable_name=", ".join(data.variables) if data.variables else "",
            vtk_classes=[
                {
                    "class": cls,
                    "variables": [lc["variable"] for lc in group if lc["class"] == cls],
                    "methods": class_methods.get(cls, []),
                }
                for cls in vtk_classes
            ],
            queries=queries,
        )

        return chunk.to_dict()

    def _collect_from_lifecycles(self, group: list[dict]) -> LifecycleData:
        """Collect statements, classes, variables, and role from lifecycle group."""
        statements: list[ast.stmt] = []
        classes: list[str] = []
        variables: list[str] = []
        role = ""

        for lifecycle in group:
            classes.append(lifecycle["class"])
            if lifecycle.get("variable"):
                variables.append(lifecycle["variable"])
            # Include property classes (e.g., vtkProperty from actor.GetProperty())
            for prop in lifecycle.get("properties", []):
                if prop["class"] and prop["class"] not in classes:
                    classes.append(prop["class"])
            statements.extend(lifecycle["statements"])
            if not role:
                role = lifecycle["type"]

        return LifecycleData(statements=statements, classes=classes, variables=variables, role=role)

    def _build_code(self, statements: list[ast.stmt], vtk_classes: list[str]) -> str:
        """Build code string from AST statements with imports and rewrites."""
        # 1. Extract source from AST statements
        code_lines = []
        for stmt in statements:
            source = ast.get_source_segment(self.code, stmt)
            if source:
                code_lines.append(source)
        raw_code = "\n".join(code_lines)

        # 2. Rewrite vtk.vtkClass() → vtkClass() and self.var → var
        rewritten = re.sub(r'\bvtk\.(vtk\w+)', r'\1', raw_code)
        rewritten = re.sub(r'\bself\.(\w+)', r'\1', rewritten)

        # 3. Add imports
        imports = self._imports(vtk_classes)
        return "\n".join(imports) + "\n\n" + rewritten

    def _determine_datatypes(self, role: str, vtk_classes: list[str]) -> tuple[str, str]:
        """Determine input/output datatypes based on role and classes."""
        if role == "renderer":
            return "vtkActor", "vtkRenderer"

        if role in ("infrastructure", "scene"):
            return "vtkRenderer", ""

        if role == "properties":
            mapper_class = next((c for c in vtk_classes if "Mapper" in c), None)
            actor_class = next((c for c in vtk_classes if "Actor" in c), None)
            if mapper_class:
                input_dt = self.mcp_client.get_class_input_datatype(mapper_class) or ""
                return input_dt, "vtkActor"
            if actor_class:
                return "vtkMapper", "vtkActor"

        # Default: use first class's datatypes
        if vtk_classes:
            input_dt = self.mcp_client.get_class_input_datatype(vtk_classes[0]) or ""
            output_dt = self.mcp_client.get_class_output_datatype(vtk_classes[0]) or ""
            return input_dt, output_dt

        return "", ""

    def _compute_visibility(self, vtk_classes: list[str]) -> float:
        """Compute average visibility score for classes."""
        if not vtk_classes:
            return 0.5

        scores = []
        for vtk_class in vtk_classes:
            score = self.mcp_client.get_class_visibility(vtk_class)
            scores.append(score if score is not None else 0.5)
        return sum(scores) / len(scores)

    def _collect_methods(
        self, group: list[dict]
    ) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
        """Collect and deduplicate methods per class from lifecycle group."""
        class_methods: dict[str, list[str]] = {}
        class_method_calls: dict[str, list[dict]] = {}

        for lifecycle in group:
            cls = lifecycle["class"]
            # Merge and dedupe methods
            merged_methods = class_methods.get(cls, []) + lifecycle.get("methods", [])
            class_methods[cls] = dedupe_preserve_order(merged_methods)
            # Merge and dedupe method_calls
            merged_calls = class_method_calls.get(cls, []) + lifecycle.get("method_calls", [])
            class_method_calls[cls] = dedupe_method_calls(merged_calls)

        return class_methods, class_method_calls

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
