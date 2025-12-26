"""
VTK API Documentation Chunker

Creates semantic chunks from vtk-python-docs.jsonl for RAG indexing.

Chunk Types:
1. Class Overview - class_doc, method list, synopsis, action_phrase
2. Constructor - vtkClass instantiation info
3. Methods - one chunk per semantic method (excludes boilerplate)
4. Inheritance - parent class

All chunks include: module, role, visibility, input_datatype, output_datatype

Code Map:
    DocChunker.__init__()
        ├── doc → module, action_phrase, synopsis, visibility_score, role, datatypes, semantic_methods
        └── _extract_own_methods()   → method docs filtered by semantic_methods

    DocChunker.extract_chunks()
        ├── _create_class_overview()    → class_overview chunk
        ├── _create_constructor_chunk() → constructor chunk
        ├── _create_method_chunks()     → method chunks (one per method)
        │       └── _create_method_chunk()
        └── _create_inheritance_chunk() → inheritance chunk
"""

from __future__ import annotations

import re
from typing import Any

from vtk_rag.mcp import VTKClient

from ..query import SemanticQuery
from .models import DocChunk


class DocChunker:
    """
    Chunks VTK API documentation into semantic units.

    Uses vtk-python-docs.jsonl fields:
    - class_name, module_name, class_doc
    - synopsis, action_phrase, visibility_score
    - role, input_datatype, output_datatype, semantic_methods
    - structured_docs.sections (method docs)

    Uses MCP for:
    - Parent class module lookup (inheritance chunk)
    """

    doc: dict[str, Any]
    class_name: str
    mcp_client: VTKClient
    module: str
    action_phrase: str
    synopsis: str
    visibility_score: float
    role: str
    input_datatype: str
    output_datatype: str
    semantic_methods: set[str]
    own_methods: dict[str, str]

    def __init__(self, doc: dict[str, Any], mcp_client: VTKClient) -> None:
        """Initialize with a single API doc record.

        Args:
            doc: Dict from vtk-python-docs.jsonl
            mcp_client: MCP client for VTK API access.
        """
        self.doc = doc
        self.class_name = doc.get('class_name', '')
        self.mcp_client = mcp_client

        # Extract metadata
        self.module = doc.get('module_name', '')
        self.action_phrase = doc.get('action_phrase', '')

        self.synopsis = doc.get('synopsis', '')

        self.visibility_score = doc.get('visibility_score', 0.3)

        self.role = doc.get('role', 'utility')
        self.input_datatype = doc.get('input_datatype', '')
        self.output_datatype = doc.get('output_datatype', '')
        self.semantic_methods = set(doc.get('semantic_methods', []))

        # Extract method docs, filtered to semantic methods only
        self.own_methods = self._extract_own_methods()

        # Query builder for semantic queries
        self.semantic_query = SemanticQuery(mcp_client)

    def extract_chunks(self) -> list[dict[str, Any]]:
        """Extract all chunks from the API doc.
        """
        if not self.class_name:
            return []

        chunks = []

        # 1. Class Overview chunk
        overview = self._create_class_overview()
        if overview:
            chunks.append(overview.to_dict())

        # 2. Constructor chunk
        constructor = self._create_constructor_chunk()
        if constructor:
            chunks.append(constructor.to_dict())

        # 3. Method chunks (one per method, excluding boilerplate)
        method_chunks = self._create_method_chunks()
        chunks.extend(c.to_dict() for c in method_chunks)

        # 4. Inheritance chunk
        inheritance = self._create_inheritance_chunk()
        if inheritance:
            chunks.append(inheritance.to_dict())

        return chunks


    def _create_class_overview(self) -> DocChunk | None:
        """Create class overview chunk."""
        class_doc = self.doc.get('class_doc', '')
        if not class_doc:
            return None

        content_parts = [f"# {self.class_name}"]
        content_parts.append("")
        content_parts.append(class_doc)
        content_parts.append("")

        # Add method list
        if self.own_methods:
            public_methods = [m for m in self.own_methods.keys() if not m.startswith('_')]
            if public_methods:
                content_parts.append("## Methods")
                content_parts.append(", ".join(sorted(public_methods)[:30]))
                if len(public_methods) > 30:
                    content_parts.append(f"... and {len(public_methods) - 30} more")

        content = "\n".join(content_parts)

        chunk_synopsis = f"Overview of {self.class_name}: {self.synopsis}"

        # Build queries: semantic query + non-semantic queries
        queries = []

        # Semantic query from action_phrase
        if self.action_phrase:
            semantic_q = self.semantic_query.class_to_query(self.action_phrase.lower())
            if semantic_q:
                queries.append(semantic_q)

        # Non-semantic queries
        non_semantic = [
            f"What is {self.class_name}?",
            f"What does {self.class_name} do?",
            f"How do I use {self.class_name}?",
        ]

        # Deduplicate while preserving order
        seen = {q.lower() for q in queries}
        for q in non_semantic:
            if q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)

        overview_action_phrase = f"overview of {self.action_phrase}" if self.action_phrase else f"{self.class_name} overview"

        return self._create_chunk(
            chunk_id=f"{self.class_name}_overview",
            chunk_type="class_overview",
            content=content,
            synopsis=chunk_synopsis,
            queries=queries,
            action_phrase=overview_action_phrase,
        )

    def _create_constructor_chunk(self) -> DocChunk | None:
        """Create constructor chunk."""
        # Look for __new__ or __init__
        constructor_doc = self.own_methods.get('__new__', '') or self.own_methods.get('__init__', '')

        # Build content
        content_parts = [f"# {self.class_name} Constructor"]
        content_parts.append("")
        var_name = self._class_to_var_name(self.class_name)
        content_parts.append(f"{var_name} = {self.class_name}()")

        if constructor_doc and 'Initialize self' not in constructor_doc:
            content_parts.append("")
            content_parts.append("## Constructor Documentation")
            content_parts.append(constructor_doc)

        content = "\n".join(content_parts)

        synopsis = self.doc.get('synopsis', '')
        chunk_synopsis = f"Create {self.class_name}"
        if synopsis:
            chunk_synopsis += f" - {synopsis}"

        queries = [
            f"How to create {self.class_name}?",
            f"{self.class_name} constructor",
        ]

        return self._create_chunk(
            chunk_id=f"{self.class_name}_constructor",
            chunk_type="constructor",
            content=content,
            synopsis=chunk_synopsis,
            queries=queries,
            action_phrase=f"{self.class_name} constructor",
        )

    def _create_method_chunks(self) -> list[DocChunk]:
        """Create a chunk for each semantic method."""
        chunks = []

        for method_name, method_doc in self.own_methods.items():
            chunk = self._create_method_chunk(method_name, method_doc)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_method_chunk(self, method_name: str, method_doc: str) -> DocChunk:
        """Create a chunk for a single method."""
        content = f"# {self.class_name}.{method_name}\n\n{method_doc.strip()}"

        description = self._extract_method_description(method_name, method_doc)
        synopsis = f"{self.class_name}.{method_name}()"
        if description:
            synopsis += f" - {description}"

        # CamelCase to words: ComputeBounds -> compute bounds
        action_phrase = f"{self.class_name} {re.sub(r'(?<!^)(?=[A-Z])', ' ', method_name).lower()}"

        # Build queries: semantic query + non-semantic queries
        queries = []

        # Semantic query from method name and class action_phrase
        if self.action_phrase:
            semantic_q = self.semantic_query.method_to_query(method_name, self.action_phrase.lower())
            if semantic_q:
                queries.append(semantic_q)

        # Non-semantic queries
        non_semantic = [
            f"How to use {self.class_name}.{method_name}",
        ]

        # Deduplicate while preserving order
        seen = {q.lower() for q in queries}
        for q in non_semantic:
            if q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)

        return self._create_chunk(
            chunk_id=f"{self.class_name}_{method_name}",
            chunk_type="method",
            content=content,
            synopsis=synopsis,
            queries=queries,
            action_phrase=action_phrase,
        )

    def _create_inheritance_chunk(self) -> DocChunk | None:
        """Create inheritance chunk showing superclass."""
        class_doc = self.doc.get('class_doc', '')

        parent_match = re.search(r'Superclass:\s*(\w+)', class_doc)
        parent_class = parent_match.group(1) if parent_match else None

        if not parent_class:
            return None

        superclass_module = self.mcp_client.get_class_module(parent_class) or ""

        content_parts = [f"# {self.class_name} Inheritance"]
        content_parts.append("")
        content_parts.append(f"**Superclass:** `{parent_class}`")
        if superclass_module:
            content_parts.append(f"**Superclass Module:** `{superclass_module}`")
        content = "\n".join(content_parts)

        chunk_synopsis = f"{self.class_name} inherits from {parent_class}"
        queries = [
            f"What does {self.class_name} inherit from?",
            f"{self.class_name} parent class",
            f"{self.class_name} base class",
        ]

        return self._create_chunk(
            chunk_id=f"{self.class_name}_inheritance",
            chunk_type="inheritance",
            content=content,
            synopsis=chunk_synopsis,
            queries=queries,
            action_phrase="",
        )

    def _create_chunk(
        self,
        chunk_id: str,
        chunk_type: str,
        content: str,
        synopsis: str,
        queries: list[str],
        action_phrase: str | None = None,
    ) -> DocChunk:
        """Create a DocChunk with common class metadata.

        Args:
            chunk_id: Unique identifier for the chunk.
            chunk_type: Type of chunk.
            content: Full text content.
            synopsis: Brief summary.
            queries: Natural language queries.
            action_phrase: Override for action_phrase (defaults to class action_phrase).

        Returns:
            DocChunk with class metadata populated.
        """
        return DocChunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            vtk_class=self.class_name,
            content=content,
            synopsis=synopsis,
            role=self.role,
            action_phrase=action_phrase if action_phrase is not None else self.action_phrase,
            visibility_score=self.visibility_score,
            module=self.module,
            input_datatype=self.input_datatype,
            output_datatype=self.output_datatype,
            queries=queries,
        )

    def _extract_own_methods(self) -> dict[str, str]:
        """Extract method docs, filtered to semantic methods only."""
        methods = {}
        sections = self.doc.get('structured_docs', {}).get('sections', {})

        for section_name, section_data in sections.items():
            if 'defined here' in section_name.lower():
                section_methods = section_data.get('methods', {})
                for name, doc in section_methods.items():
                    # Only include if it's a semantic method (not boilerplate)
                    if not self.semantic_methods or name in self.semantic_methods:
                        methods[name] = doc

        return methods

    def _class_to_var_name(self, class_name: str) -> str:
        """Convert class name to variable: vtkSphereSource -> sphereSource, vtkmDataSet -> dataSet."""
        if class_name.startswith('vtkm'):
            name = class_name[4:]
        elif class_name.startswith('vtk'):
            name = class_name[3:]
        else:
            name = class_name
        return name[0].lower() + name[1:] if name else class_name.lower()

    def _extract_method_description(self, method_name: str, method_doc: str) -> str:
        """Extract description from method doc (last non-signature line)."""
        lines = [line for line in method_doc.strip().split('\n') if line.strip()]
        for line in reversed(lines):
            if not line.startswith(method_name) and '->' not in line:
                return line.strip()
        return ""

