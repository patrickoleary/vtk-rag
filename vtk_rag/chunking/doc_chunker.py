#!/usr/bin/env python3
"""
VTK API Documentation Chunker

Creates semantic chunks from vtk-python-docs.jsonl for RAG indexing.

Chunk Types:
1. Class Overview - class_doc, method list, synopsis, action_phrase
2. Constructor - vtkClass + instantiation info
3. Property Groups - related methods grouped by property name:
   - Set/Get pairs with optional Min/Max bounds
   - On/Off toggles with Set/Get
   - Add/Remove pairs with RemoveAll/GetNumberOf
   - Enable/Disable pairs with Set/Get
4. Standalone Methods - methods not fitting property group patterns
5. Inheritance - parent class chain

All chunks include: module, role, visibility, input_datatype, output_datatype
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .doc_chunk import DocChunk
from .doc_query_generator import DocQueryGenerator
from .vtk_categories import BOILERPLATE_METHODS


class DocChunker:
    """
    Chunks VTK API documentation into semantic units.

    Extracts from vtk-python-docs.jsonl which contains:
    - class_name, module_name, class_doc
    - synopsis, action_phrase, role, visibility
    - structured_docs with method sections
    """

    def __init__(self, doc: dict[str, Any]) -> None:
        """Initialize with a single API doc record.

        Args:
            doc: Dict from vtk-python-docs.jsonl
        """
        self.doc = doc
        self.class_name = doc.get('class_name', '')

        # Query generator
        self.query_generator = DocQueryGenerator()

        # Extract common metadata and methods upfront
        self.base_metadata = self._extract_base_metadata()
        self.own_methods = self._extract_own_methods()

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

        # 3. Property Group chunks (returns used method names)
        property_chunks, used_methods = self._create_property_chunks()
        chunks.extend(c.to_dict() for c in property_chunks)

        # 4. Standalone Method chunks (methods not in property groups)
        standalone_chunks = self._create_standalone_method_chunks(used_methods)
        chunks.extend(c.to_dict() for c in standalone_chunks)

        # 5. Inheritance chunk
        inheritance = self._create_inheritance_chunk()
        if inheritance:
            chunks.append(inheritance.to_dict())

        return chunks

    def _extract_base_metadata(self) -> dict[str, Any]:
        """Extract common metadata from doc."""
        module = self.doc.get('module_name', '')

        # Get input/output datatypes from method signatures
        input_dt, output_dt = self._extract_datatypes()

        return {
            "class_name": self.class_name,
            "module": module,
            "role": self.doc.get('role', ''),
            "visibility": self.doc.get('visibility', ''),
            "synopsis": self.doc.get('synopsis', ''),
            "action_phrase": self.doc.get('action_phrase', ''),
            "input_datatype": input_dt,
            "output_datatype": output_dt,
        }

    def _extract_datatypes(self) -> tuple[str, str]:
        """Extract input/output datatypes from method signatures."""
        input_datatype = "N/A"
        output_datatype = "N/A"

        sections = self.doc.get('structured_docs', {}).get('sections', {})

        for section_data in sections.values():
            methods = section_data.get('methods', {})

            # Look for GetOutput to determine output type
            for method_name in ['GetOutput', 'GetOutputPort']:
                if method_name in methods:
                    method_doc = methods[method_name]
                    # Parse return type: GetOutput(self) -> vtkPolyData
                    match = re.search(r'->\s*(vtk\w+)', method_doc)
                    if match:
                        output_datatype = match.group(1)
                        break

            # Look for SetInputData/GetInput to determine input type
            for method_name in ['SetInputData', 'GetInput', 'SetInput']:
                if method_name in methods:
                    method_doc = methods[method_name]
                    # Parse parameter type: SetInputData(self, __a:vtkDataSet)
                    match = re.search(r':\s*(vtk\w+)', method_doc)
                    if match:
                        input_datatype = match.group(1)
                        break
                    # Or return type for GetInput
                    match = re.search(r'->\s*(vtk\w+)', method_doc)
                    if match:
                        input_datatype = match.group(1)
                        break

        return input_datatype, output_datatype

    def _extract_own_methods(self) -> dict[str, str]:
        """Extract only methods defined on this class (not inherited)."""
        methods = {}
        sections = self.doc.get('structured_docs', {}).get('sections', {})

        for section_name, section_data in sections.items():
            # Only include "Methods defined here" and "Static methods defined here"
            if 'defined here' in section_name.lower():
                section_methods = section_data.get('methods', {})
                methods.update(section_methods)

        return methods

    def _create_class_overview(self) -> DocChunk | None:
        """Create class overview chunk."""
        class_doc = self.doc.get('class_doc', '')
        synopsis = self.doc.get('synopsis', '')
        action_phrase = self.doc.get('action_phrase', '')
        module = self.doc.get('module_name', '')

        if not class_doc:
            return None

        # Build content
        content_parts = [f"# {self.class_name}"]
        content_parts.append("")
        content_parts.append(f"**Module:** `{module}`")
        content_parts.append(f"**Role:** {self.base_metadata['role']}")
        content_parts.append(f"**Visibility:** {self.base_metadata['visibility']}")
        content_parts.append("")
        content_parts.append("## Description")
        content_parts.append(class_doc)
        content_parts.append("")

        # Add method list (just names)
        if self.own_methods:
            # Filter out dunder methods for the list
            public_methods = [m for m in self.own_methods.keys() if not m.startswith('_')]
            if public_methods:
                content_parts.append("## Methods")
                content_parts.append(", ".join(sorted(public_methods)[:30]))  # Limit to 30
                if len(public_methods) > 30:
                    content_parts.append(f"... and {len(public_methods) - 30} more")

        content = "\n".join(content_parts)

        # Build synopsis for RAG matching
        chunk_synopsis = f"{action_phrase.capitalize() if action_phrase else self.class_name}: {synopsis}"

        # Generate queries
        queries = self.query_generator.generate_queries(
            chunk_type="class_overview",
            class_name=self.class_name,
            action_phrase=action_phrase,
        )

        return DocChunk(
            chunk_id=f"{self.class_name}_overview",
            chunk_type="class_overview",
            class_name=self.class_name,
            content=content,
            synopsis=chunk_synopsis,
            role=self.base_metadata.get('role', ''),
            action_phrase=self.base_metadata.get('action_phrase', ''),
            visibility=self.base_metadata.get('visibility', ''),
            metadata=self.base_metadata.copy(),
            queries=queries,
        )

    def _create_constructor_chunk(self) -> DocChunk | None:
        """Create constructor chunk."""
        synopsis = self.doc.get('synopsis', '')
        action_phrase = self.doc.get('action_phrase', '')

        # Look for __new__ or __init__
        constructor_doc = self.own_methods.get('__new__', '') or self.own_methods.get('__init__', '')

        # Build content
        content_parts = [f"# {self.class_name} Constructor"]
        content_parts.append("")
        content_parts.append(f"**Module:** `{self.base_metadata['module']}`")
        content_parts.append("")
        content_parts.append("## Instantiation")
        content_parts.append("```python")
        content_parts.append(f"from {self.base_metadata['module']} import {self.class_name}")
        content_parts.append(f"obj = {self.class_name}()")
        content_parts.append("```")
        content_parts.append("")

        if constructor_doc and 'Initialize self' not in constructor_doc:
            content_parts.append("## Constructor Documentation")
            content_parts.append(constructor_doc)

        if synopsis:
            content_parts.append("")
            content_parts.append(f"**Synopsis:** {synopsis}")

        content = "\n".join(content_parts)

        # Synopsis for RAG
        chunk_synopsis = f"Create {self.class_name}: {action_phrase}" if action_phrase else f"Instantiate {self.class_name}"

        # Generate queries
        queries = self.query_generator.generate_queries(
            chunk_type="constructor",
            class_name=self.class_name,
            action_phrase=action_phrase,
        )

        return DocChunk(
            chunk_id=f"{self.class_name}_constructor",
            chunk_type="constructor",
            class_name=self.class_name,
            content=content,
            synopsis=chunk_synopsis,
            role=self.base_metadata.get('role', ''),
            action_phrase=self.base_metadata.get('action_phrase', ''),
            visibility=self.base_metadata.get('visibility', ''),
            metadata=self.base_metadata.copy(),
            queries=queries,
        )

    def _create_property_chunks(self) -> tuple[list[DocChunk], set]:
        """Create property group chunks (Set/Get pairs).

        Returns:
            Tuple of (chunks, used_method_names)
        """
        chunks = []

        # Find all Set* methods and group with their Get* counterparts
        property_groups, used_methods = self._group_methods_by_property()

        for prop_name, prop_methods in property_groups.items():
            chunk = self._create_single_property_chunk(prop_name, prop_methods)
            if chunk:
                chunks.append(chunk)

        return chunks, used_methods

    def _group_methods_by_property(self) -> tuple[dict[str, dict[str, str]], set]:
        """Group methods by property name (Set/Get/Min/Max).

        Returns:
            Tuple of (property_groups, used_method_names)
        """
        groups = defaultdict(dict)
        used_methods = set()

        for method_name in self.own_methods:
            if method_name in used_methods:
                continue

            # Check for Set* pattern
            if method_name.startswith('Set') and len(method_name) > 3:
                prop_name = method_name[3:]  # e.g., "Radius" from "SetRadius"

                # Skip if prop_name starts with lowercase (not a property setter)
                if prop_name and prop_name[0].isupper():
                    related = {
                        f'Set{prop_name}': self.own_methods.get(f'Set{prop_name}', ''),
                        f'Get{prop_name}': self.own_methods.get(f'Get{prop_name}', ''),
                        f'Get{prop_name}MinValue': self.own_methods.get(f'Get{prop_name}MinValue', ''),
                        f'Get{prop_name}MaxValue': self.own_methods.get(f'Get{prop_name}MaxValue', ''),
                    }
                    # Filter out empty ones
                    related = {k: v for k, v in related.items() if v}

                    if related:
                        groups[prop_name] = related
                        used_methods.update(related.keys())

            # Check for *On/*Off pattern
            elif method_name.endswith('On') and len(method_name) > 2:
                prop_name = method_name[:-2]  # e.g., "GenerateNormals" from "GenerateNormalsOn"

                if prop_name and prop_name[0].isupper():
                    related = {
                        f'{prop_name}On': self.own_methods.get(f'{prop_name}On', ''),
                        f'{prop_name}Off': self.own_methods.get(f'{prop_name}Off', ''),
                        f'Set{prop_name}': self.own_methods.get(f'Set{prop_name}', ''),
                        f'Get{prop_name}': self.own_methods.get(f'Get{prop_name}', ''),
                    }
                    related = {k: v for k, v in related.items() if v}

                    if related and prop_name not in groups:
                        groups[prop_name] = related
                        used_methods.update(related.keys())

            # Check for Add* pattern
            elif method_name.startswith('Add') and len(method_name) > 3:
                prop_name = method_name[3:]  # e.g., "Actor" from "AddActor"

                if prop_name and prop_name[0].isupper():
                    related = {
                        f'Add{prop_name}': self.own_methods.get(f'Add{prop_name}', ''),
                        f'Remove{prop_name}': self.own_methods.get(f'Remove{prop_name}', ''),
                        f'RemoveAll{prop_name}s': self.own_methods.get(f'RemoveAll{prop_name}s', ''),
                        f'GetNumberOf{prop_name}s': self.own_methods.get(f'GetNumberOf{prop_name}s', ''),
                    }
                    related = {k: v for k, v in related.items() if v}

                    if related and prop_name not in groups:
                        groups[prop_name] = related
                        used_methods.update(related.keys())

            # Check for Enable* pattern
            elif method_name.startswith('Enable') and len(method_name) > 6:
                prop_name = method_name[6:]  # e.g., "Feature" from "EnableFeature"

                if prop_name and prop_name[0].isupper():
                    related = {
                        f'Enable{prop_name}': self.own_methods.get(f'Enable{prop_name}', ''),
                        f'Disable{prop_name}': self.own_methods.get(f'Disable{prop_name}', ''),
                        f'Set{prop_name}': self.own_methods.get(f'Set{prop_name}', ''),
                        f'Get{prop_name}': self.own_methods.get(f'Get{prop_name}', ''),
                    }
                    related = {k: v for k, v in related.items() if v}

                    if related and prop_name not in groups:
                        groups[prop_name] = related
                        used_methods.update(related.keys())

        return dict(groups), used_methods

    def _create_single_property_chunk(
        self,
        prop_name: str,
        prop_methods: dict[str, str],
    ) -> DocChunk | None:
        """Create a chunk for a single property group."""
        if not prop_methods:
            return None

        # Build content
        content_parts = [f"# {self.class_name}: {prop_name}"]
        content_parts.append("")
        content_parts.append(f"**Module:** `{self.base_metadata['module']}`")
        content_parts.append(f"**Role:** {self.base_metadata['role']}")
        content_parts.append(f"**Visibility:** {self.base_metadata['visibility']}")
        content_parts.append("")

        # Add each method in the group
        for method_name, method_doc in sorted(prop_methods.items()):
            content_parts.append(f"## {method_name}")
            content_parts.append("")
            # Clean up method doc
            content_parts.append(method_doc.strip())
            content_parts.append("")

        content = "\n".join(content_parts)

        # Build synopsis
        # Convert CamelCase to words: ThetaResolution -> theta resolution
        readable_prop = re.sub(r'(?<!^)(?=[A-Z])', ' ', prop_name).lower()
        method_types = []
        if any('Set' in m for m in prop_methods):
            method_types.append('set')
        if any('Get' in m and 'Min' not in m and 'Max' not in m for m in prop_methods):
            method_types.append('get')
        if any('Min' in m or 'Max' in m for m in prop_methods):
            method_types.append('bounds')
        if any('On' in m or 'Off' in m for m in prop_methods):
            method_types.append('toggle')
        if any('Add' in m for m in prop_methods):
            method_types.append('add')
        if any('Remove' in m for m in prop_methods):
            method_types.append('remove')
        if any('Enable' in m for m in prop_methods):
            method_types.append('enable')
        if any('Disable' in m for m in prop_methods):
            method_types.append('disable')

        chunk_synopsis = f"{self.class_name} {readable_prop} property ({', '.join(method_types)})"

        # Metadata with property info
        metadata = self.base_metadata.copy()
        metadata['property_name'] = prop_name
        metadata['method_names'] = list(prop_methods.keys())

        # Generate queries
        queries = self.query_generator.generate_queries(
            chunk_type="property_group",
            class_name=self.class_name,
            action_phrase=self.base_metadata.get('action_phrase', ''),
            properties=[prop_name],
        )

        return DocChunk(
            chunk_id=f"{self.class_name}_{prop_name}",
            chunk_type="property_group",
            class_name=self.class_name,
            content=content,
            synopsis=chunk_synopsis,
            role=self.base_metadata.get('role', ''),
            action_phrase=self.base_metadata.get('action_phrase', ''),
            visibility=self.base_metadata.get('visibility', ''),
            metadata=metadata,
            queries=queries,
        )

    def _create_standalone_method_chunks(self, used_methods: set) -> list[DocChunk]:
        """Create chunks for methods not captured by property groups.

        These include:
        - Action methods: Render, Update, Clear, ShallowCopy
        - Add/Remove methods: AddActor, RemoveActor
        - Read-only getters: GetBounds, GetInput, GetOutput
        - Query methods: HasOpaqueGeometry, IsActiveCameraCreated
        - Coordinate transforms: WorldToView, ViewToWorld
        """
        chunks = []

        # Find uncaptured methods (excluding boilerplate)
        uncaptured = {
            name: doc for name, doc in self.own_methods.items()
            if name not in used_methods and name not in BOILERPLATE_METHODS
        }

        if not uncaptured:
            return []

        # Group related methods together
        method_groups = self._group_standalone_methods(uncaptured)

        for group_name, group_methods in method_groups.items():
            chunk = self._create_method_group_chunk(group_name, group_methods)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _group_standalone_methods(self, methods: dict[str, str]) -> dict[str, dict[str, str]]:
        """Group standalone methods by category."""
        groups = defaultdict(dict)
        used = set()

        # Group Add/Remove pairs
        for name in methods:
            if name.startswith('Add') and len(name) > 3:
                thing = name[3:]
                related = {}
                for prefix in ['Add', 'Remove', 'RemoveAll', 'GetNumberOf', 'Get']:
                    for suffix in [thing, thing + 's', thing.rstrip('s')]:
                        key = f'{prefix}{suffix}'
                        if key in methods and key not in used:
                            related[key] = methods[key]
                            used.add(key)
                if related:
                    groups[f'{thing}_collection'] = related

        # Group coordinate transforms
        coord_methods = {}
        for name in methods:
            if name not in used and any(x in name for x in ['ToWorld', 'ToView', 'ToPose', 'ToDisplay']):
                coord_methods[name] = methods[name]
                used.add(name)
        if coord_methods:
            groups['coordinate_transforms'] = coord_methods

        # Group render methods
        render_methods = {}
        for name in methods:
            if name not in used and 'Render' in name:
                render_methods[name] = methods[name]
                used.add(name)
        if render_methods:
            groups['rendering'] = render_methods

        # Remaining methods go into individual chunks or small groups
        remaining = {n: d for n, d in methods.items() if n not in used}

        # Group remaining by first verb/prefix
        prefix_groups = defaultdict(dict)
        for name, doc in remaining.items():
            # Extract prefix (Get, Has, Is, Create, Make, etc.)
            prefix = None
            for p in ['Get', 'Has', 'Is', 'Create', 'Make', 'Compute', 'Release', 'Capture']:
                if name.startswith(p):
                    prefix = p
                    break

            if prefix:
                prefix_groups[f'{prefix.lower()}_methods'][name] = doc
            else:
                # Individual action methods
                prefix_groups[f'{name}_method'][name] = doc

        groups.update(prefix_groups)

        return dict(groups)

    def _create_method_group_chunk(
        self,
        group_name: str,
        group_methods: dict[str, str],
    ) -> DocChunk | None:
        """Create a chunk for a group of standalone methods."""
        if not group_methods:
            return None

        # Build content
        content_parts = [f"# {self.class_name}: {group_name.replace('_', ' ').title()}"]
        content_parts.append("")
        content_parts.append(f"**Module:** `{self.base_metadata['module']}`")
        content_parts.append(f"**Role:** {self.base_metadata['role']}")
        content_parts.append(f"**Visibility:** {self.base_metadata['visibility']}")
        content_parts.append("")

        for method_name, method_doc in sorted(group_methods.items()):
            content_parts.append(f"## {method_name}")
            content_parts.append("")
            content_parts.append(method_doc.strip())
            content_parts.append("")

        content = "\n".join(content_parts)

        # Build synopsis
        method_names = list(group_methods.keys())
        if len(method_names) == 1:
            chunk_synopsis = f"{self.class_name}.{method_names[0]}()"
        else:
            chunk_synopsis = f"{self.class_name} {group_name.replace('_', ' ')}: {', '.join(method_names[:3])}"
            if len(method_names) > 3:
                chunk_synopsis += f" (+{len(method_names) - 3} more)"

        # Metadata
        metadata = self.base_metadata.copy()
        metadata['method_names'] = method_names
        metadata['group_name'] = group_name

        # Generate queries
        queries = self.query_generator.generate_queries(
            chunk_type="standalone_methods",
            class_name=self.class_name,
            action_phrase=self.base_metadata.get('action_phrase', ''),
            methods=method_names,
        )

        return DocChunk(
            chunk_id=f"{self.class_name}_{group_name}",
            chunk_type="standalone_methods",
            class_name=self.class_name,
            content=content,
            synopsis=chunk_synopsis,
            role=self.base_metadata.get('role', ''),
            action_phrase=self.base_metadata.get('action_phrase', ''),
            visibility=self.base_metadata.get('visibility', ''),
            metadata=metadata,
            queries=queries,
        )

    def _create_inheritance_chunk(self) -> DocChunk | None:
        """Create inheritance chunk showing parent class chain."""
        class_doc = self.doc.get('class_doc', '')

        # Extract parent class from class_doc
        # Pattern: "Superclass: vtkPolyDataAlgorithm"
        parent_match = re.search(r'Superclass:\s*(\w+)', class_doc)
        parent_class = parent_match.group(1) if parent_match else None

        if not parent_class:
            return None

        # Get inherited method sections
        sections = self.doc.get('structured_docs', {}).get('sections', {})
        inherited_sections = []

        for section_name, section_data in sections.items():
            if 'inherited from' in section_name.lower():
                # Extract the parent class name from section
                parent_match = re.search(r'inherited from\s+(\S+)', section_name, re.IGNORECASE)
                if parent_match:
                    inherited_from = parent_match.group(1).rstrip(':')
                    method_count = section_data.get('method_count', 0)
                    inherited_sections.append((inherited_from, method_count))

        # Build content
        content_parts = [f"# {self.class_name} Inheritance"]
        content_parts.append("")
        content_parts.append(f"**Module:** `{self.base_metadata['module']}`")
        content_parts.append(f"**Superclass:** `{parent_class}`")
        content_parts.append("")
        content_parts.append("## Inheritance Chain")
        content_parts.append("")

        if inherited_sections:
            for parent, count in inherited_sections:
                content_parts.append(f"- **{parent}**: {count} methods")

        content = "\n".join(content_parts)

        # Synopsis
        chunk_synopsis = f"{self.class_name} inherits from {parent_class}"

        # Metadata
        metadata = self.base_metadata.copy()
        metadata['parent_class'] = parent_class
        metadata['inherited_from'] = [p for p, _ in inherited_sections]

        # Generate queries
        queries = self.query_generator.generate_queries(
            chunk_type="inheritance",
            class_name=self.class_name,
            action_phrase="",
        )

        return DocChunk(
            chunk_id=f"{self.class_name}_inheritance",
            chunk_type="inheritance",
            class_name=self.class_name,
            content=content,
            synopsis=chunk_synopsis,
            role=self.base_metadata.get('role', ''),
            action_phrase=self.base_metadata.get('action_phrase', ''),
            visibility=self.base_metadata.get('visibility', ''),
            metadata=metadata,
            queries=queries,
        )
