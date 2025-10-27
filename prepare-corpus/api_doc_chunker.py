#!/usr/bin/env python3
"""
API Documentation Chunker for VTK RAG System

Chunks API documentation (vtk-python-docs.jsonl) into method-level chunks.
Creates small, focused reference chunks (150-500 tokens).

Part of redesigned chunking strategy - separates API_DOC from examples.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for API doc chunking"""
    target_tokens: int = 300       # Sweet spot for API reference
    min_tokens: int = 150          # Minimum useful method doc
    max_tokens: int = 500          # Hard limit
    chars_per_token: float = 4.0   # Approximation


@dataclass
class ApiDocChunk:
    """Represents an API documentation chunk"""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content: str  # API reference text
    content_type: str = "api_doc"
    metadata: Dict[str, Any] = None
    source_type: str = "api_doc"
    original_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ApiDocParser:
    """Parse VTK API documentation structure"""
    
    @staticmethod
    def parse_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse API doc structure
        Returns: {class_name, module, class_doc, methods}
        """
        class_name = doc.get('class_name', 'Unknown')
        module = doc.get('module_name', '')
        
        # Get structured docs if available
        structured_docs = doc.get('structured_docs', {})
        
        # Extract class description
        class_doc = doc.get('class_doc', '') or structured_docs.get('description', '')
        
        # Extract methods from sections
        methods = {}
        sections = structured_docs.get('sections', {})
        
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict) and 'methods' in section_data:
                methods.update(section_data['methods'])
        
        # Get full content as fallback
        full_content = doc.get('content', '')
        
        return {
            'class_name': class_name,
            'module': module,
            'class_doc': class_doc,
            'methods': methods,
            'full_content': full_content,
            'structured_docs': structured_docs
        }
    
    @staticmethod
    def extract_inheritance(class_doc: str) -> Optional[str]:
        """Extract parent class from documentation"""
        # Look for patterns like "Inherits from X" or "Subclass of X"
        patterns = [
            r'[Ii]nherits?\s+from\s+(\w+)',
            r'[Ss]ubclass\s+of\s+(\w+)',
            r'[Dd]erived\s+from\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, class_doc)
            if match:
                return match.group(1)
        
        return None


class MethodExtractor:
    """Extract and parse method information"""
    
    @staticmethod
    def parse_method_signature(method_doc: str) -> Dict[str, Any]:
        """
        Parse method signature to extract parameters and return type
        Returns: {method_name, parameters, return_type, description}
        """
        lines = method_doc.split('\n')
        
        # First line usually has signature
        signature_line = lines[0] if lines else ""
        
        # Extract method name
        method_match = re.match(r'###?\s*(\w+)\s*\(', signature_line)
        method_name = method_match.group(1) if method_match else "unknown"
        
        # Extract parameters (simplified)
        param_match = re.search(r'\((.*?)\)', signature_line)
        parameters = []
        if param_match:
            param_str = param_match.group(1)
            if param_str.strip():
                # Split by comma, clean up
                parameters = [p.strip() for p in param_str.split(',') if p.strip()]
        
        # Extract description (rest of lines)
        description = '\n'.join(lines[1:]).strip()
        
        return {
            'method_name': method_name,
            'parameters': parameters,
            'description': description
        }
    
    @staticmethod
    def group_methods(methods: Dict[str, str], max_methods_per_chunk: int = 3) -> List[List[Tuple[str, str]]]:
        """
        Group methods into chunks
        Returns: [[('method1', 'doc1'), ...], ...]
        """
        method_items = list(methods.items())
        groups = []
        
        for i in range(0, len(method_items), max_methods_per_chunk):
            group = method_items[i:i + max_methods_per_chunk]
            groups.append(group)
        
        return groups


class ApiDocChunker:
    """
    Main chunker for API documentation
    
    Strategy:
    1. Class overview chunk (class description + inheritance)
    2. Method chunks (group 1-3 related methods)
    
    Features:
    - Method-level granularity
    - Small chunks (150-500 tokens) for precise retrieval
    - Metadata: class_name, module, method_name
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.parser = ApiDocParser()
        self.method_extractor = MethodExtractor()
    
    def chunk_api_doc(self, doc: Dict[str, Any]) -> List[ApiDocChunk]:
        """
        Chunk an API documentation entry
        
        Args:
            doc: API doc dict with 'class_name', 'module_name', 'content', etc.
        
        Returns:
            List of ApiDocChunk objects
        """
        # Parse doc structure
        parsed = self.parser.parse_doc(doc)
        
        class_name = parsed['class_name']
        module = parsed['module']
        class_doc = parsed['class_doc']
        methods = parsed['methods']
        
        if not class_name or class_name == 'Unknown':
            logger.warning("No class_name in API doc")
            return []
        
        chunks = []
        
        # Chunk 1: Class overview
        if class_doc:
            overview_chunk = self._create_class_overview_chunk(
                class_name,
                module,
                class_doc,
                len(chunks)
            )
            chunks.append(overview_chunk)
        
        # Chunk 2+: Methods
        if methods:
            method_chunks = self._create_method_chunks(
                class_name,
                module,
                methods,
                len(chunks)
            )
            chunks.extend(method_chunks)
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _create_class_overview_chunk(
        self,
        class_name: str,
        module: str,
        class_doc: str,
        start_index: int
    ) -> ApiDocChunk:
        """Create a class overview chunk"""
        
        # Build content
        content_parts = [f"# {class_name}"]
        
        if module:
            content_parts.append(f"**Module:** `{module}`")
            content_parts.append("")
        
        # Extract inheritance
        inheritance = self.parser.extract_inheritance(class_doc)
        if inheritance:
            content_parts.append(f"**Inherits from:** {inheritance}")
            content_parts.append("")
        
        # Add description
        content_parts.append("## Description")
        content_parts.append(class_doc)
        
        content = '\n'.join(content_parts)
        
        # Build metadata
        metadata = {
            'class_name': class_name,
            'module': module,
            'chunk_type': 'class_overview',
            'inheritance': inheritance
        }
        
        chunk_id = f"{class_name}_overview"
        
        return ApiDocChunk(
            chunk_id=chunk_id,
            chunk_index=start_index,
            total_chunks=0,  # Will update later
            content=content,
            content_type="api_doc",
            metadata=metadata,
            source_type="api_doc",
            original_id=class_name
        )
    
    def _create_method_chunks(
        self,
        class_name: str,
        module: str,
        methods: Dict[str, str],
        start_index: int
    ) -> List[ApiDocChunk]:
        """Create method chunks (group 1-3 methods per chunk)"""
        
        chunks = []
        
        # Group methods
        method_groups = self.method_extractor.group_methods(
            methods,
            max_methods_per_chunk=3
        )
        
        for group_idx, method_group in enumerate(method_groups):
            chunk = self._create_method_group_chunk(
                class_name,
                module,
                method_group,
                start_index + group_idx
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_method_group_chunk(
        self,
        class_name: str,
        module: str,
        method_group: List[Tuple[str, str]],
        chunk_index: int
    ) -> ApiDocChunk:
        """Create a chunk from a group of methods"""
        
        # Build content
        content_parts = [f"# {class_name}"]
        
        if module:
            content_parts.append(f"**Module:** `{module}`")
            content_parts.append("")
        
        content_parts.append("## Methods")
        content_parts.append("")
        
        method_names = []
        
        for method_name, method_doc in method_group:
            method_names.append(method_name)
            
            # Parse method
            parsed = self.method_extractor.parse_method_signature(method_doc)
            
            # Add method section
            content_parts.append(f"### {method_name}")
            
            # Add parameters if any
            if parsed['parameters']:
                params_str = ', '.join(parsed['parameters'])
                content_parts.append(f"**Parameters:** `{params_str}`")
            
            # Add description
            if parsed['description']:
                content_parts.append("")
                content_parts.append(parsed['description'])
            
            content_parts.append("")  # Blank line between methods
        
        content = '\n'.join(content_parts)
        
        # Build metadata
        metadata = {
            'class_name': class_name,
            'module': module,
            'chunk_type': 'methods',
            'method_names': method_names,
            'method_count': len(method_names)
        }
        
        # Chunk ID
        if len(method_names) == 1:
            chunk_id = f"{class_name}_{method_names[0]}"
        else:
            chunk_id = f"{class_name}_methods_{chunk_index}"
        
        return ApiDocChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=0,  # Will update later
            content=content,
            content_type="api_doc",
            metadata=metadata,
            source_type="api_doc",
            original_id=class_name
        )


def main():
    """Test the API doc chunker"""
    print("=" * 80)
    print("ApiDocChunker - Test")
    print("=" * 80)
    
    # Test with mock API doc
    test_doc = {
        'class_name': 'vtkCylinderSource',
        'module_name': 'vtkmodules.vtkFiltersSources',
        'class_doc': 'Generate a polygonal cylinder centered at the origin. The cylinder is oriented along the y-axis.',
        'structured_docs': {
            'description': 'Generate a polygonal cylinder centered at the origin.',
            'sections': {
                'Methods defined here': {
                    'methods': {
                        'SetRadius': 'SetRadius(float)\n\nSet the radius of the cylinder. Default is 0.5.',
                        'GetRadius': 'GetRadius() -> float\n\nGet the radius of the cylinder.',
                        'SetHeight': 'SetHeight(float)\n\nSet the height of the cylinder. Default is 1.0.',
                        'GetHeight': 'GetHeight() -> float\n\nGet the height of the cylinder.',
                        'SetResolution': 'SetResolution(int)\n\nSet the number of facets used to define the cylinder. Default is 6.',
                        'GetResolution': 'GetResolution() -> int\n\nGet the number of facets.'
                    }
                }
            }
        }
    }
    
    # Run chunker
    chunker = ApiDocChunker()
    chunks = chunker.chunk_api_doc(test_doc)
    
    print(f"\nGenerated {len(chunks)} chunk(s)")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Tokens: ~{len(chunk.content) // 4}")
        print(f"Metadata:")
        print(f"  class_name: {chunk.metadata['class_name']}")
        print(f"  module: {chunk.metadata['module']}")
        print(f"  chunk_type: {chunk.metadata['chunk_type']}")
        
        if 'method_names' in chunk.metadata:
            print(f"  method_names: {chunk.metadata['method_names']}")
        
        print(f"\nContent preview (first 400 chars):")
        print(chunk.content[:400])
        print("...")
        print()
    
    print("=" * 80)
    print("✓ ApiDocChunker test complete")


if __name__ == '__main__':
    main()
