#!/usr/bin/env python3
"""
Explanation-Only Chunker for VTK RAG System

Extracts descriptive text from examples/docs, strips code blocks,
generates explanation-focused chunks (300-700 tokens).

Part of redesigned chunking strategy - separates EXPLANATION from CODE.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for explanation chunking"""
    target_tokens: int = 500      # Sweet spot for explanations
    min_tokens: int = 300          # Minimum meaningful explanation
    max_tokens: int = 700          # Hard limit
    chars_per_token: float = 4.0   # Approximation


@dataclass
class ExplanationChunk:
    """Represents an explanation-only chunk"""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content: str  # Text only, no code
    content_type: str = "explanation"
    metadata: Dict[str, Any] = None
    source_type: str = "example"
    original_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextExtractor:
    """Extract and clean explanatory text"""
    
    @staticmethod
    def strip_code_blocks(text: str) -> str:
        """Remove all code blocks (```...```)"""
        # Remove fenced code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code (but keep text)
        # text = re.sub(r'`([^`]+)`', r'\1', text)  # Keep content, remove backticks
        
        return text
    
    @staticmethod
    def extract_title_and_description(example: Dict[str, Any]) -> tuple[str, str]:
        """Extract title and main description"""
        title = example.get('title', 'Untitled')
        
        # Try different fields for description
        description = (
            example.get('explanation', '') or
            example.get('description', '') or
            example.get('class_doc', '') or
            ''
        )
        
        return title, description
    
    @staticmethod
    def extract_sections(text: str) -> List[tuple[str, str]]:
        """
        Extract sections from markdown text
        Returns: [(section_name, content), ...]
        """
        sections = []
        
        # Split by headers (## or ###)
        parts = re.split(r'^(#{2,3})\s+(.+)$', text, flags=re.MULTILINE)
        
        if len(parts) == 1:
            # No sections, return as single section
            return [('Main', parts[0].strip())]
        
        # Parse sections
        current_section = 'Introduction'
        current_content = parts[0].strip() if parts[0].strip() else None
        
        if current_content:
            sections.append((current_section, current_content))
        
        i = 1
        while i < len(parts):
            if i + 2 < len(parts):
                # parts[i] is ##, parts[i+1] is section name, parts[i+2] is content
                section_name = parts[i + 1].strip()
                content = parts[i + 2].strip()
                
                if content:
                    sections.append((section_name, content))
                
                i += 3
            else:
                break
        
        return sections


class ConceptExtractor:
    """Extract VTK concepts from text"""
    
    # VTK concept keywords
    CONCEPTS = [
        'rendering', 'visualization', 'geometry', 'filtering', 
        'pipeline', 'mapper', 'actor', 'source', 'filter',
        'polydata', 'imagedata', 'volume', 'graphics',
        'dataset', 'points', 'cells', 'scalars', 'vectors',
        'camera', 'light', 'texture', 'transform',
        'reader', 'writer', 'importer', 'exporter'
    ]
    
    @staticmethod
    def extract_concepts(text: str, vtk_classes: List[str] = None) -> List[str]:
        """Extract key VTK concepts from text"""
        text_lower = text.lower()
        concepts = []
        
        # Extract from keyword list
        for concept in ConceptExtractor.CONCEPTS:
            if concept in text_lower:
                concepts.append(concept)
        
        # Extract from VTK class names
        if vtk_classes:
            for cls in vtk_classes:
                # Extract concept from class name
                # e.g., vtkCylinderSource -> sources
                if 'Source' in cls:
                    concepts.append('sources')
                elif 'Filter' in cls:
                    concepts.append('filters')
                elif 'Mapper' in cls:
                    concepts.append('mappers')
                elif 'Actor' in cls:
                    concepts.append('actors')
                elif 'Renderer' in cls:
                    concepts.append('rendering')
                elif 'Reader' in cls or 'Writer' in cls:
                    concepts.append('io')
        
        return sorted(list(set(concepts)))
    
    @staticmethod
    def extract_vtk_classes_from_text(text: str) -> List[str]:
        """Extract VTK class names mentioned in text"""
        # Pattern: vtk followed by uppercase letter and word chars
        pattern = r'\b(vtk[A-Z]\w+)\b'
        matches = re.findall(pattern, text)
        return sorted(list(set(matches)))


class TopicTypeDetector:
    """Detect the type of explanation"""
    
    @staticmethod
    def detect_topic_type(title: str, text: str, source_type: str) -> str:
        """
        Detect topic type
        Returns: 'tutorial', 'concept', 'api_doc', or 'reference'
        """
        title_lower = title.lower()
        text_lower = text.lower()
        
        # API documentation
        if source_type == 'api_doc':
            return 'api_doc'
        
        # Tutorial indicators
        tutorial_words = ['example', 'how to', 'tutorial', 'guide', 'step by step']
        if any(word in title_lower or word in text_lower for word in tutorial_words):
            return 'tutorial'
        
        # Concept indicators
        concept_words = ['introduction', 'overview', 'concept', 'understanding', 'what is']
        if any(word in title_lower or word in text_lower for word in concept_words):
            return 'concept'
        
        # Default
        return 'reference'


class ExplanationChunker:
    """
    Main chunker for creating explanation-only chunks
    
    Features:
    - Extracts text only (strips code blocks)
    - Extracts concepts from text
    - Detects topic type (tutorial/concept/api_doc)
    - Links to related code chunks
    - Target: 300-700 tokens per chunk
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.text_extractor = TextExtractor()
        self.concept_extractor = ConceptExtractor()
        self.topic_detector = TopicTypeDetector()
    
    def chunk_example(
        self, 
        example: Dict[str, Any],
        code_queries: List[str] = None
    ) -> List[ExplanationChunk]:
        """
        Chunk an example into explanation-only chunks
        
        Args:
            example: Dict with 'title', 'explanation', 'category', etc.
            code_queries: Optional list of code queries to transform
        
        Returns:
            List of ExplanationChunk objects
        """
        example_id = example.get('id', 'unknown')
        title, description = self.text_extractor.extract_title_and_description(example)
        category = example.get('category', '')
        
        if not description:
            logger.warning(f"No description in example {example_id}")
            return []
        
        # Strip code blocks from description
        text_only = self.text_extractor.strip_code_blocks(description)
        
        if not text_only.strip():
            logger.warning(f"No text after stripping code: {example_id}")
            return []
        
        # Extract metadata
        vtk_classes = self.concept_extractor.extract_vtk_classes_from_text(text_only)
        concepts = self.concept_extractor.extract_concepts(text_only, vtk_classes)
        topic_type = self.topic_detector.detect_topic_type(
            title, text_only, example.get('source_type', 'example')
        )
        
        # Generate explanation queries (basic version, Phase 5 will enhance)
        explanation_queries = self._generate_basic_explanation_queries(
            title, vtk_classes, code_queries or []
        )
        
        # Chunk the text
        chunks = self._chunk_text(
            text_only,
            example_id,
            title,
            category,
            explanation_queries,
            concepts,
            topic_type,
            vtk_classes
        )
        
        return chunks
    
    def _generate_basic_explanation_queries(
        self,
        title: str,
        vtk_classes: List[str],
        code_queries: List[str]
    ) -> List[str]:
        """
        Generate basic explanation queries
        Phase 5 (QueryGenerator) will provide more sophisticated generation
        """
        queries = []
        
        # Transform code queries
        transformations = {
            "How do I": "What is",
            "Show me code to": "Explain how to",
            "code for": "concept of",
            "example": "explanation",
            "Create": "Understanding",
            "Write": "Learn about"
        }
        
        for query in code_queries:
            transformed = query
            for old, new in transformations.items():
                transformed = transformed.replace(old, new)
            if transformed != query:
                queries.append(transformed)
        
        # Add class-based queries
        for cls in vtk_classes[:2]:  # Limit to top 2
            queries.append(f"What is {cls}?")
            queries.append(f"Explain {cls}")
        
        # Add title-based query
        queries.append(f"Explain {title}")
        
        return list(set(queries))  # Deduplicate
    
    def _chunk_text(
        self,
        text: str,
        example_id: str,
        title: str,
        category: str,
        explanation_queries: List[str],
        concepts: List[str],
        topic_type: str,
        vtk_classes: List[str]
    ) -> List[ExplanationChunk]:
        """
        Split text into chunks of 300-700 tokens
        """
        text = text.strip()
        total_chars = len(text)
        estimated_tokens = int(total_chars / self.config.chars_per_token)
        
        # Single chunk if small enough
        if estimated_tokens <= self.config.max_tokens:
            return [self._create_chunk(
                text,
                example_id,
                title,
                category,
                0,
                1,
                explanation_queries,
                concepts,
                topic_type,
                vtk_classes
            )]
        
        # Multiple chunks: split by sections or paragraphs
        sections = self.text_extractor.extract_sections(text)
        
        if len(sections) > 1:
            # Split by sections
            return self._chunk_by_sections(
                sections,
                example_id,
                title,
                category,
                explanation_queries,
                concepts,
                topic_type,
                vtk_classes
            )
        else:
            # Split by paragraphs
            return self._chunk_by_paragraphs(
                text,
                example_id,
                title,
                category,
                explanation_queries,
                concepts,
                topic_type,
                vtk_classes
            )
    
    def _chunk_by_sections(
        self,
        sections: List[tuple[str, str]],
        example_id: str,
        title: str,
        category: str,
        explanation_queries: List[str],
        concepts: List[str],
        topic_type: str,
        vtk_classes: List[str]
    ) -> List[ExplanationChunk]:
        """Chunk by markdown sections"""
        chunks = []
        current_sections = []
        current_size = 0
        
        for section_name, content in sections:
            section_text = f"## {section_name}\n\n{content}"
            section_size = len(section_text)
            
            if current_size + section_size > self.config.max_tokens * self.config.chars_per_token:
                # Create chunk from current sections
                if current_sections:
                    chunk_text = '\n\n'.join(current_sections)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        example_id,
                        title,
                        category,
                        len(chunks),
                        0,
                        explanation_queries,
                        concepts,
                        topic_type,
                        vtk_classes
                    ))
                
                # Start new chunk
                current_sections = [section_text]
                current_size = section_size
            else:
                current_sections.append(section_text)
                current_size += section_size
        
        # Final chunk
        if current_sections:
            chunk_text = '\n\n'.join(current_sections)
            chunks.append(self._create_chunk(
                chunk_text,
                example_id,
                title,
                category,
                len(chunks),
                0,
                explanation_queries,
                concepts,
                topic_type,
                vtk_classes
            ))
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _chunk_by_paragraphs(
        self,
        text: str,
        example_id: str,
        title: str,
        category: str,
        explanation_queries: List[str],
        concepts: List[str],
        topic_type: str,
        vtk_classes: List[str]
    ) -> List[ExplanationChunk]:
        """Chunk by paragraphs when no sections"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_paras = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.config.max_tokens * self.config.chars_per_token:
                if current_paras:
                    chunk_text = '\n\n'.join(current_paras)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        example_id,
                        title,
                        category,
                        len(chunks),
                        0,
                        explanation_queries,
                        concepts,
                        topic_type,
                        vtk_classes
                    ))
                
                current_paras = [para]
                current_size = para_size
            else:
                current_paras.append(para)
                current_size += para_size
        
        # Final chunk
        if current_paras:
            chunk_text = '\n\n'.join(current_paras)
            chunks.append(self._create_chunk(
                chunk_text,
                example_id,
                title,
                category,
                len(chunks),
                0,
                explanation_queries,
                concepts,
                topic_type,
                vtk_classes
            ))
        
        # Update total
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        example_id: str,
        title: str,
        category: str,
        chunk_index: int,
        total_chunks: int,
        explanation_queries: List[str],
        concepts: List[str],
        topic_type: str,
        vtk_classes: List[str]
    ) -> ExplanationChunk:
        """Create an ExplanationChunk object"""
        chunk_id = f"{title}_explanation_{chunk_index}"
        
        # Build metadata
        metadata = {
            'title': title,
            'category': category,
            'explanation_queries': explanation_queries,
            'concepts': concepts,
            'topic_type': topic_type,
            'vtk_classes': vtk_classes,
            'related_code_chunk': f"{title}Pythonic_code_0",  # Prefer Pythonic
            'related_image_chunk': f"{title}_image_0" if 'image' in example_id.lower() else None
        }
        
        return ExplanationChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=text,
            content_type="explanation",
            metadata=metadata,
            source_type="example",
            original_id=example_id
        )


def main():
    """Test the explanation chunker"""
    print("=" * 80)
    print("ExplanationChunker - Test")
    print("=" * 80)
    
    # Test example
    test_example = {
        'id': 'CylinderExample',
        'title': 'Cylinder Example',
        'category': 'Geometric Objects',
        'explanation': '''# Creating a 3D Cylinder

This example demonstrates how to create and render a 3D cylinder using VTK.

## Explanation

The vtkCylinderSource class creates a polygonal representation of a cylinder. 
The cylinder is centered at the origin and aligned along the y-axis by default.

## Key Classes

- **vtkCylinderSource**: Generates the cylinder geometry
- **vtkPolyDataMapper**: Maps the geometry to graphics primitives  
- **vtkActor**: Represents the cylinder in the scene

## Parameters

You can control the cylinder with these methods:

- SetRadius(float): Controls the radius of the cylinder
- SetHeight(float): Controls the height along the y-axis
- SetResolution(int): Controls the number of facets (higher = smoother)

```python
cylinder = vtkCylinderSource()
cylinder.SetRadius(1.0)
cylinder.SetHeight(2.0)
```

## Rendering

To render the cylinder, you need to create a complete visualization pipeline
with a mapper, actor, renderer, and render window.
''',
        'code_queries': [
            'How do I create a cylinder in VTK?',
            'Show me code to render a cylinder'
        ]
    }
    
    # Run chunker
    chunker = ExplanationChunker()
    chunks = chunker.chunk_example(test_example, test_example['code_queries'])
    
    print(f"\nGenerated {len(chunks)} chunk(s)")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Tokens: ~{len(chunk.content) // 4}")
        print(f"Metadata:")
        print(f"  topic_type: {chunk.metadata['topic_type']}")
        print(f"  concepts: {chunk.metadata['concepts']}")
        print(f"  vtk_classes: {chunk.metadata['vtk_classes']}")
        print(f"  explanation_queries (first 3):")
        for q in chunk.metadata['explanation_queries'][:3]:
            print(f"    - {q}")
        print(f"\nText preview (first 400 chars):")
        print(chunk.content[:400])
        print("...")
        print()
    
    print("=" * 80)
    print("✓ ExplanationChunker test complete")


if __name__ == '__main__':
    main()
