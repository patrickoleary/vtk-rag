"""Filter builders for Qdrant queries.

Provides a fluent interface for building Qdrant filter conditions
to narrow search results by metadata fields.
"""

from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)


class FilterBuilder:
    """Build Qdrant filter conditions with a fluent interface.

    Filters narrow search results by metadata fields. Qdrant uses boolean
    logic with three condition types:

    - **must**: All conditions must match (AND logic)
    - **should**: At least one condition should match (OR logic)
    - **must_not**: No conditions may match (exclusion)

    Available Fields:
        Code collection (vtk_code):
            - type: Chunk type (Visualization Pipeline, Rendering Infrastructure, vtkmodules.*)
            - vtk_class: Primary VTK class name
            - function_name: Containing function name
            - roles: Functional roles (source_geometric, filter_general, etc.)
            - input_datatype: Input data type (vtkPolyData, vtkImageData, etc.)
            - output_datatype: Output data type
            - visibility_score: User-facing likelihood (0.0-1.0)
            - example_id: Source example URL
            - variable_name: Primary variable name

        Doc collection (vtk_docs):
            - chunk_type: class_overview, constructor, property_group, standalone_methods, inheritance
            - class_name: VTK class name
            - role: Functional role
            - visibility: User-facing likelihood (very_likely, likely, maybe, etc.)
            - metadata.module: VTK module path
            - metadata.input_datatype: Input data type
            - metadata.output_datatype: Output data type

    Methods:
        match(field, value): Exact match (must) - all must match
        match_any(field, values): Match any value in list (must)
        range(field, gt, gte, lt, lte): Numeric range (must)
        exclude(field, value): Exclusion (must_not) - none may match
        should_match(field, value): Optional match (should) - at least one should match
        build(): Returns Qdrant Filter object
        from_dict(filters): Create from dict (class method)

    Example - Fluent Builder:
        # Full control with all condition types
        filters = (
            FilterBuilder()
            .match("role", "source_geometric")           # must match exactly
            .match_any("vtk_class", ["vtkSphereSource", "vtkConeSource"])  # must match one
            .range("visibility_score", gte=0.7)          # must be >= 0.7
            .exclude("chunk_type", "inheritance")        # must NOT match
            .should_match("type", "Visualization Pipeline")  # bonus if matches
            .build()
        )

    Example - Dict Shorthand:
        # Simpler syntax, but only supports must conditions (no exclude/should)
        filters = {
            "role": "source_geometric",                  # exact match
            "vtk_class": ["vtkSphereSource", "vtkConeSource"],  # match any
            "visibility_score": {"gte": 0.7},            # range
        }
        # Equivalent to:
        # FilterBuilder().match("role", ...).match_any("vtk_class", ...).range(...).build()

        # Note: Dict shorthand does NOT support:
        # - exclude() / must_not conditions
        # - should_match() / should conditions
        # Use fluent builder for those.
    """

    def __init__(self) -> None:
        """Initialize empty filter builder."""
        self._must: list[FieldCondition] = []
        self._should: list[FieldCondition] = []
        self._must_not: list[FieldCondition] = []

    def match(self, field: str, value: Any) -> "FilterBuilder":
        """Add exact match condition (must).

        Args:
            field: Field name in payload.
            value: Value to match exactly.

        Returns:
            Self for chaining.
        """
        self._must.append(
            FieldCondition(key=field, match=MatchValue(value=value))
        )
        return self

    def match_any(self, field: str, values: list[Any]) -> "FilterBuilder":
        """Add match-any condition (must match one of values).

        Args:
            field: Field name in payload.
            values: List of values to match.

        Returns:
            Self for chaining.
        """
        self._must.append(
            FieldCondition(key=field, match=MatchAny(any=values))
        )
        return self

    def range(
        self,
        field: str,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
    ) -> "FilterBuilder":
        """Add range condition for numeric fields.

        Args:
            field: Field name in payload.
            gt: Greater than.
            gte: Greater than or equal.
            lt: Less than.
            lte: Less than or equal.

        Returns:
            Self for chaining.
        """
        self._must.append(
            FieldCondition(key=field, range=Range(gt=gt, gte=gte, lt=lt, lte=lte))
        )
        return self

    def exclude(self, field: str, value: Any) -> "FilterBuilder":
        """Add exclusion condition (must not match).

        Args:
            field: Field name in payload.
            value: Value to exclude.

        Returns:
            Self for chaining.
        """
        self._must_not.append(
            FieldCondition(key=field, match=MatchValue(value=value))
        )
        return self

    def should_match(self, field: str, value: Any) -> "FilterBuilder":
        """Add optional match condition (should).

        Args:
            field: Field name in payload.
            value: Value to optionally match.

        Returns:
            Self for chaining.
        """
        self._should.append(
            FieldCondition(key=field, match=MatchValue(value=value))
        )
        return self

    def build(self) -> Filter | None:
        """Build the Qdrant Filter object.

        Returns:
            Filter object, or None if no conditions added.
        """
        if not self._must and not self._should and not self._must_not:
            return None

        return Filter(
            must=self._must if self._must else None,
            should=self._should if self._should else None,
            must_not=self._must_not if self._must_not else None,
        )

    @classmethod
    def from_dict(cls, filters: dict[str, Any]) -> "FilterBuilder":
        """Create FilterBuilder from a dictionary.

        Supports simple key-value pairs and range dicts:
            {"role": "source_geometric"}  # exact match
            {"visibility_score": {"gte": 0.7}}  # range
            {"class_name": ["vtkA", "vtkB"]}  # match any

        Args:
            filters: Dictionary of filter conditions.

        Returns:
            FilterBuilder instance.
        """
        builder = cls()

        for field, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                builder.range(
                    field,
                    gt=value.get("gt"),
                    gte=value.get("gte"),
                    lt=value.get("lt"),
                    lte=value.get("lte"),
                )
            elif isinstance(value, list):
                # Match any
                builder.match_any(field, value)
            else:
                # Exact match
                builder.match(field, value)

        return builder
