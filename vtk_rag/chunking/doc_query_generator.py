"""Query generator for VTK API documentation chunks."""

from __future__ import annotations

import re


class DocQueryGenerator:
    """
    Generates natural language queries for API documentation chunks.

    Uses:
    - action_phrase for class-level context
    - camelCase→words for method/property names (like code chunker)
    """

    def generate_queries(
        self,
        chunk_type: str,
        class_name: str,
        action_phrase: str,
        methods: list[str] | None = None,
        properties: list[str] | None = None,
    ) -> list[str]:
        """Generate natural language queries based on chunk type.

        Args:
            chunk_type: Type of chunk ('class_overview', 'constructor',
                       'property_group', 'standalone_methods', 'inheritance').
            class_name: VTK class name (e.g., 'vtkSphereSource').
            action_phrase: Action phrase for the class (e.g., 'create a sphere').
            methods: Method names for standalone_methods chunks.
            properties: Property names for property_group chunks.

        Returns:
            List of deduplicated natural language queries for RAG retrieval.
        """
        queries = []

        if chunk_type == "class_overview":
            queries = self._overview_queries(class_name, action_phrase)
        elif chunk_type == "constructor":
            queries = self._constructor_queries(class_name, action_phrase)
        elif chunk_type == "property_group":
            queries = self._property_queries(class_name, action_phrase, properties or [])
        elif chunk_type == "standalone_methods":
            queries = self._method_queries(class_name, action_phrase, methods or [])
        elif chunk_type == "inheritance":
            queries = self._inheritance_queries(class_name)

        return self._dedupe(queries)

    def _overview_queries(self, class_name: str, action_phrase: str) -> list[str]:
        """Queries for class overview chunks."""
        queries = [
            f"What is {class_name}?",
            f"What does {class_name} do?",
            f"{class_name} documentation",
        ]

        if action_phrase:
            queries.extend([
                f"How do I {action_phrase}?",
                f"VTK {action_phrase}",
                f"VTK class for {action_phrase}",
            ])

        return queries

    def _constructor_queries(self, class_name: str, action_phrase: str) -> list[str]:
        """Queries for constructor chunks."""
        queries = [
            f"How to create {class_name}?",
            f"How to instantiate {class_name}?",
            f"{class_name} constructor",
        ]

        if action_phrase:
            queries.append(f"Create {action_phrase}")

        return queries

    def _property_queries(
        self,
        class_name: str,
        action_phrase: str,
        properties: list[str],
    ) -> list[str]:
        """Queries for property group chunks."""
        queries = []

        for prop in properties:
            prop_readable = self._camel_to_words(prop)
            queries.extend([
                f"How to set {prop_readable} in {class_name}?",
                f"How to set {prop_readable} in VTK?",
                f"{class_name} {prop_readable}",
            ])

            if action_phrase:
                queries.append(f"Set {prop_readable} for {action_phrase}")

        return queries

    def _method_queries(
        self,
        class_name: str,
        action_phrase: str,
        methods: list[str],
    ) -> list[str]:
        """Queries for standalone method chunks."""
        queries = []

        for method in methods[:5]:  # Limit
            method_readable = self._camel_to_words(method)
            queries.extend([
                f"How to {method_readable} in VTK?",
                f"{class_name} {method_readable}",
            ])

            if action_phrase:
                queries.append(f"How to {method_readable} for {action_phrase}?")

        return queries

    def _inheritance_queries(self, class_name: str) -> list[str]:
        """Queries for inheritance chunks."""
        return [
            f"What does {class_name} inherit from?",
            f"{class_name} parent class",
            f"{class_name} base class",
        ]

    def _camel_to_words(self, name: str) -> str:
        """Convert CamelCase to lowercase words."""
        words = re.sub(r'([A-Z])', r' \1', name).strip().lower()
        return words

    def _dedupe(self, queries: list[str]) -> list[str]:
        """Remove duplicate queries."""
        seen: set[str] = set()
        result = []
        for q in queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                result.append(q)
        return result
