"""Semantic query generation for VTK code chunks.

Generates natural language "how do you" queries from VTK class action phrases
and method calls.

Used by:
    - ../doc/chunker.py (DocChunker)
    - ../code/semantic_chunk.py (SemanticChunk)

Code Map:
    SemanticQuery
        build()                      # public API for multiple classes
            └── build_queries()      # single class queries
                    ├── class_to_query()     # action phrase → query
                    └── method_to_query()    # method name → query
                            ├── _to_words()      # CamelCase → words
                            └── _with_article()  # add a/an
        function_name_to_query()     # helper function name → query
"""

import re

from vtk_rag.mcp import VTKClient


class SemanticQuery:
    """Build semantic queries from VTK classes and method calls."""

    mcp_client: VTKClient

    def __init__(self, mcp_client: VTKClient) -> None:
        self.mcp_client = mcp_client

    # Map gerund/noun endings to verbs for class-level queries
    ENDING_TO_VERB = {
        # Core actions
        "creation": "create",
        "generation": "generate",
        "reading": "read",
        "writing": "write",
        "rendering": "render",
        "display": "display",
        "visualization": "visualize",
        # Data operations
        "computation": "compute",
        "calculation": "calculate",
        "estimation": "estimate",
        "evaluation": "evaluate",
        "analysis": "analyze",
        "comparison": "compare",
        # Data transformation
        "extraction": "extract",
        "conversion": "convert",
        "transformation": "transform",
        "interpolation": "interpolate",
        "resampling": "resample",
        "smoothing": "smooth",
        "filtering": "filter",
        "clipping": "clip",
        "cutting": "cut",
        "merging": "merge",
        "partitioning": "partition",
        "tessellation": "tessellate",
        "deformation": "deform",
        # Data management
        "manipulation": "manipulate",
        "representation": "represent",
        "management": "manage",
        "storage": "store",
        "removal": "remove",
        "processing": "process",
        "handling": "handle",
        "grouping": "group",
        # Mapping and location
        "mapping": "map",
        "placement": "place",
        "location": "locate",
        "probing": "probe",
        # Traversal and iteration
        "iteration": "iterate",
        "traversal": "traverse",
        "selection": "select",
        "picking": "pick",
        # Drawing and plotting
        "drawing": "draw",
        "plotting": "plot",
        # Other
        "exporting": "export",
        "importing": "import",
        "encoding": "encode",
        "decoding": "decode",
        "serialization": "serialize",
        "deserialization": "deserialize",
        "compression": "compress",
        "integration": "integrate",
        "synchronization": "synchronize",
        "communication": "communicate",
        "execution": "execute",
        "interaction": "interact with",
        "definition": "define",
        "specification": "specify",
        "implementation": "implement",
        "construction": "construct",
        "modification": "modify",
        "adjustment": "adjust",
        "alignment": "align",
        "orientation": "orient",
        "positioning": "position",
        "layout": "layout",
        "access": "access",
        "control": "control",
        # Geometric operations
        "extrusion": "extrude",
        "decimation": "decimate",
        "reduction": "reduce",
        "subdivision": "subdivide",
        "triangulation": "triangulate",
        "simplification": "simplify",
        "thresholding": "threshold",
        "slicing": "slice",
        "splitting": "split",
        "casting": "cast",
        # Data operations
        "sorting": "sort",
        "collection": "collect",
        "aggregation": "aggregate",
        "distribution": "distribute",
        "decomposition": "decompose",
        "appending": "append",
        "caching": "cache",
        "sampling": "sample",
        "tracking": "track",
        "detection": "detect",
        "measurement": "measure",
        "assignment": "assign",
        "parsing": "parse",
        "reflection": "reflect",
    }

    # Mass nouns that don't take articles
    MASS_NOUNS = {
        "data", "information", "output", "input", "metadata",
        "geometry", "content", "storage", "memory", "hardware", "software",
    }

    # Lifecycle methods to skip (don't describe configuration)
    LIFECYCLE_METHODS = {
        "Update", "Initialize", "Render", "Start", "Modified", "Delete",
        "ShallowCopy", "DeepCopy", "Execute", "Finalize", "Allocate", "Release",
    }

    # Method prefix patterns for query generation
    METHOD_PREFIXES = {
        "Set": ("set the", "of"),
        "Add": ("add a", "to"),
        "Remove": ("remove a", "from"),
        "Create": ("create a", "for"),
        "Clear": ("clear the", "of"),
        "Reset": ("reset the", "of"),
        "Build": ("build the", "of"),
        "Compute": ("compute the", "of"),
    }

    # Method suffix patterns for query generation
    METHOD_SUFFIXES = {
        "On": "enable",
        "Off": "disable",
    }

    def build(
        self,
        vtk_classes: list[str],
        class_method_calls: dict[str, list[dict]],
    ) -> list[str]:
        """Build queries for multiple VTK classes and their method calls.

        Args:
            vtk_classes: List of VTK class names
            class_method_calls: Dict mapping class names to method call dicts

        Returns:
            List of deduplicated "how do you" queries
        """
        all_queries = []

        for vtk_class in vtk_classes:
            # Get action phrase (e.g., "polygonal sphere creation")
            action_phrase = self.mcp_client.get_class_action_phrase(vtk_class)
            if action_phrase:
                action_phrase = action_phrase[0].upper() + action_phrase[1:]
            else:
                # Fallback: vtkEventDataDevice -> "Event data device"
                name = vtk_class[3:] if vtk_class.startswith("vtk") else vtk_class
                action_phrase = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).capitalize()

            # Get method calls for this class
            method_calls = class_method_calls.get(vtk_class, [])[:8]

            # Generate queries for this class
            queries = self.build_queries(action_phrase, method_calls)
            all_queries.extend(queries)

        # Deduplicate queries while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries

    def build_queries(
        self, action_phrase: str, method_calls: list[dict]
    ) -> list[str]:
        """Build queries for a VTK class from its action phrase and method calls.

        Args:
            action_phrase: The class action phrase (e.g., "polygonal sphere creation")
            method_calls: List of method call dicts with "name" and optional "args"

        Returns:
            List of deduplicated "how do you" queries
        """
        queries = []

        # Class-level query
        class_query = self.class_to_query(action_phrase.lower())
        if class_query:
            queries.append(class_query)

        # Method-level queries
        for method_call in method_calls:
            query = self.method_to_query(method_call["name"], action_phrase.lower())
            if query:
                queries.append(query)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)

        return unique

    def class_to_query(self, action_phrase: str) -> str | None:
        """Convert a class action_phrase to a 'how do you' query.

        Parses the action_phrase ending to determine the verb and subject.

        Args:
            action_phrase: Class action phrase (e.g., "isosurface generation")

        Returns:
            Query string or None if action_phrase is empty

        Examples:
            "isosurface generation" -> "How do you generate an isosurface?"
            "data reading" -> "How do you read data?"
            "image to structured grid conversion" -> "How do you convert an image to a structured grid?"
            "2D chart rendering" -> "How do you render a 2D chart?"
        """
        if not action_phrase:
            return None

        words = action_phrase.split()
        if not words:
            return None

        last_word = words[-1]
        if last_word in self.ENDING_TO_VERB:
            verb = self.ENDING_TO_VERB[last_word]
            subject = " ".join(words[:-1])
            if subject:
                first_word = subject.split()[0].lower()
                if first_word in self.MASS_NOUNS or first_word in {"a", "an", "the"}:
                    return f"How do you {verb} {subject}?"
                article = "an" if subject[0].lower() in "aeiou" else "a"
                return f"How do you {verb} {article} {subject}?"
            return f"How do you {verb}?"

        # No recognized ending - use the phrase as-is with generic verb
        return f"How do you use {action_phrase}?"

    def method_to_query(self, method_name: str, action_phrase: str) -> str | None:
        """Convert a VTK method call to a 'how do you' query with class context.

        Args:
            method_name: VTK method name (e.g., "SetRadius")
            action_phrase: Class action phrase for context (e.g., "sphere")

        Returns:
            Query string or None for getters/lifecycle methods

        Examples:
            SetRadius, "sphere" -> "How do you set the radius of a sphere?"
            AddRenderer, "render window" -> "How do you add a renderer to a render window?"
            VisibilityOn, "actor" -> "How do you enable visibility on an actor?"
        """
        # Skip getters and lifecycle methods
        if method_name.startswith("Get"):
            return None
        if method_name in self.LIFECYCLE_METHODS:
            return None

        class_ref = self._with_article(action_phrase)

        # Prefix patterns -> "How do you [verb] the [thing] of a [class]?"
        for prefix, (verb, prep) in self.METHOD_PREFIXES.items():
            if method_name.startswith(prefix):
                thing = self._to_words(method_name[len(prefix):])
                if thing:
                    return f"How do you {verb} {thing} {prep} {class_ref}?"
                return f"How do you {verb[:-4]} {class_ref}?"  # Remove " the"

        # Suffix patterns -> "How do you enable/disable [thing] on a [class]?"
        for suffix, verb in self.METHOD_SUFFIXES.items():
            if method_name.endswith(suffix):
                thing = self._to_words(method_name[:-len(suffix)])
                return f"How do you {verb} {thing} on {class_ref}?"

        # Other methods
        return f"How do you {self._to_words(method_name)} {class_ref}?"

    def _to_words(self, name: str) -> str:
        """Convert CamelCase to lowercase words.

        Args:
            name: CamelCase string (e.g., "SetRadius")

        Returns:
            Lowercase words (e.g., "set radius")
        """
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()

    def _with_article(self, subject: str) -> str:
        """Add appropriate article (a/an) to subject.

        Skips article for mass nouns and subjects already having an article.

        Args:
            subject: Subject phrase (e.g., "sphere", "data")

        Returns:
            Subject with article (e.g., "a sphere", "data")
        """
        first_word = subject.split()[0].lower()
        if first_word in self.MASS_NOUNS or first_word in {"a", "an", "the"}:
            return subject
        article = "an" if subject[0].lower() in "aeiou" else "a"
        return f"{article} {subject}"

    # Action prefixes for helper function name query generation
    FUNCTION_ACTION_PREFIXES = {
        "make", "create", "build", "setup", "get", "generate", "compute", "calculate",
        "init", "initialize", "load", "read", "write", "save", "render", "draw",
        "add", "set", "update", "convert", "transform", "apply", "process", "extract", "filter"
    }

    def function_name_to_query(self, func_name: str) -> str:
        """Convert a helper function name to a semantic query.

        Transforms function names like 'make_hexahedron', 'makeHexahedron', or 'create_lookup_table'
        into natural language queries like 'how to make a hexahedron'.

        For single-word names or unrecognized prefixes, falls back to a generic query.

        Args:
            func_name: The helper function name (e.g., 'make_hexahedron' or 'makeHexahedron')

        Returns:
            A semantic query string.
        """
        # Split on underscores first (snake_case)
        parts = [p for p in func_name.split("_") if p]

        # If only one part, try splitting on camelCase
        if len(parts) == 1:
            # Split camelCase: makeHexahedron -> ['make', 'Hexahedron']
            parts = re.findall(r'[a-z]+|[A-Z][a-z]*', func_name)

        # Single-word name: "helper" -> "how to use helper"
        if len(parts) < 2:
            return f"how to use {func_name}"

        # Check for common action prefixes
        if parts[0].lower() not in self.FUNCTION_ACTION_PREFIXES:
            # Unrecognized prefix: "my_helper" -> "how to use my helper"
            subject = " ".join(p.lower() for p in parts)
            return f"how to use {subject}"

        action = parts[0].lower()
        # Join remaining parts with spaces, lowercase: makeHexahedron -> "hexahedron"
        subject = " ".join(p.lower() for p in parts[1:])

        # Add article for readability
        article = "an" if subject[0].lower() in "aeiou" else "a"

        return f"how to {action} {article} {subject}"
