#!/usr/bin/env python3
"""
Query Generator for VTK Code Chunks

Generates natural language queries from code chunks using:
1. Pattern templates for composite chunk types (Rendering Infrastructure, Visualization Pipeline)
2. Configuration categories detected from method calls (camera, lighting, background, etc.)
3. Specific queries with actual parameter values from code
4. Class-based queries (e.g., "How to use vtkSphereSource?")
"""

import re
from typing import Any

from .vtk_query_patterns import PATTERN_QUERIES, QUERY_CATEGORIES


class CodeQueryGenerator:
    """
    Generates natural language queries from VTK code chunks.

    Uses pattern templates, configuration detection, and actual parameter values
    to create queries that match how users would search for VTK functionality.
    """

    def __init__(self):
        """Initialize query generator."""

    def generate_queries(self, chunk: dict[str, Any]) -> list[str]:
        """
        Generate queries for a code chunk.

        Args:
            chunk: Code chunk dictionary with type, metadata, synopsis, etc.

        Returns:
            List of natural language queries
        """
        queries = []
        chunk_type = chunk.get('type', '')
        vtk_classes = chunk.get('metadata', {}).get('vtk_classes', [])
        synopsis = chunk.get('synopsis', '')

        # 1. Pattern-level queries for composite chunk types
        pattern_queries = self._get_pattern_queries(chunk_type)
        queries.extend(pattern_queries)

        # 2. Configuration-specific queries based on detected methods
        config_queries = self._get_config_queries(vtk_classes)
        queries.extend(config_queries)

        # 3. Specific queries with actual parameter values
        value_queries = self._get_value_queries(vtk_classes, synopsis)
        queries.extend(value_queries)

        # 4. Class-based queries
        class_queries = self._get_class_queries(vtk_classes)
        queries.extend(class_queries)

        # Deduplicate and clean
        queries = self._dedupe_queries(queries)

        return queries

    def _get_pattern_queries(self, chunk_type: str) -> list[str]:
        """Get pattern-level queries for chunk type."""
        return PATTERN_QUERIES.get(chunk_type, [])

    def _get_config_queries(self, vtk_classes: list[dict[str, Any]]) -> list[str]:
        """Get configuration-specific queries based on detected methods."""
        queries = []
        detected_configs: set[str] = set()

        for cls_info in vtk_classes:
            class_name = cls_info.get('class', '')
            methods = cls_info.get('methods', [])

            for config_name, config in QUERY_CATEGORIES.items():
                # Check if class matches
                if class_name in config.classes:
                    detected_configs.add(config_name)
                    continue

                # Check if methods match (but only for non-class-specific configs)
                if not config.classes:  # Generic configs like background, window
                    for method in methods:
                        if method in config.methods:
                            detected_configs.add(config_name)
                            break
                elif class_name in config.classes:
                    # Class-specific config, add queries
                    detected_configs.add(config_name)

        # Add queries for detected configurations
        for config_name in detected_configs:
            config = QUERY_CATEGORIES[config_name]
            queries.extend(config.queries)

        return queries

    def _get_value_queries(
        self,
        vtk_classes: list[dict[str, Any]],
        synopsis: str
    ) -> list[str]:
        """Generate queries with actual parameter values from code."""
        queries = []

        # Parse synopsis for "X set to Y" patterns
        # Synopsis format: "action phrase with prop set to value, prop2 set to value2"
        set_to_pattern = re.compile(r'(\w+(?:\s+\w+)*)\s+set\s+to\s+([^,|]+)')

        for match in set_to_pattern.finditer(synopsis):
            prop_name = match.group(1).strip()
            value = match.group(2).strip()

            # Skip generic/variable values
            if self._is_variable_value(value):
                # Still generate query without specific value
                queries.append(f"How do I set {prop_name} in VTK?")
            else:
                # Generate query with specific value
                queries.append(f"How do I set {prop_name} to {value} in VTK?")
                queries.append(f"VTK {prop_name} {value}")

        # Parse for "X added" patterns
        added_pattern = re.compile(r'(\w+(?:\s+\w+)*)\s+added')
        for match in added_pattern.finditer(synopsis):
            thing = match.group(1).strip()
            queries.append(f"How do I add {thing} in VTK?")

        # Parse for "X enabled/disabled" patterns
        enabled_pattern = re.compile(r'(\w+(?:\s+\w+)*)\s+(enabled|disabled)')
        for match in enabled_pattern.finditer(synopsis):
            feature = match.group(1).strip()
            state = match.group(2)
            if state == "enabled":
                queries.append(f"How do I enable {feature} in VTK?")
            else:
                queries.append(f"How do I disable {feature} in VTK?")

        return queries

    def _is_variable_value(self, value: str) -> bool:
        """Check if value is a variable reference rather than literal."""
        # Variable patterns: starts with lowercase, contains dots, function calls
        if not value:
            return True

        # Literal numbers
        if re.match(r'^-?\d+\.?\d*$', value):
            return False

        # Tuple of numbers
        if re.match(r'^\([\d\s,.-]+\)$', value):
            return False

        # String literals
        if value.startswith("'") or value.startswith('"'):
            return False

        # Color lookups are useful
        if 'GetColor' in value or 'Color3d' in value:
            return False

        # Everything else is probably a variable
        return True

    def _get_class_queries(self, vtk_classes: list[dict[str, Any]]) -> list[str]:
        """Generate class-based queries."""
        queries = []

        for cls_info in vtk_classes:
            class_name = cls_info.get('class', '')
            if not class_name.startswith('vtk'):
                continue

            # Basic class queries
            queries.append(f"How to use {class_name}?")
            queries.append(f"{class_name} example")

        return queries

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        """Deduplicate and clean queries."""
        seen = set()
        result = []

        for q in queries:
            # Normalize for comparison
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                result.append(q)

        return result
