#!/usr/bin/env python3
"""
Image Chunker for VTK RAG System

Creates metadata-only chunks for images with bidirectional links
to related code and explanation chunks.

Part of redesigned chunking strategy - separates IMAGE metadata.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageChunk:
    """Represents an image metadata chunk"""
    chunk_id: str
    chunk_index: int = 0
    total_chunks: int = 1
    content: str = ""  # Empty for images (just metadata)
    content_type: str = "image"
    metadata: Dict[str, Any] = None
    source_type: str = "example"
    original_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImageTypeDetector:
    """Detect type of image"""
    
    @staticmethod
    def detect_image_type(example: Dict[str, Any]) -> str:
        """
        Detect image type
        Returns: 'result', 'diagram', 'baseline', or 'screenshot'
        """
        example_id = example.get('id', '').lower()
        title = example.get('title', '').lower()
        
        # Check for baseline (test expected output)
        if 'baseline' in example_id or 'baseline' in title:
            return 'baseline'
        
        # Check for diagram
        if 'diagram' in example_id or 'diagram' in title or 'schematic' in title:
            return 'diagram'
        
        # Check for screenshot
        if 'screenshot' in example_id or 'screenshot' in title:
            return 'screenshot'
        
        # Default: result image (visualization output)
        return 'result'


class VisualConceptExtractor:
    """Extract concepts shown in image"""
    
    # Visual concepts that might be in images
    VISUAL_CONCEPTS = [
        '3d rendering', '2d rendering', 'visualization',
        'geometry', 'mesh', 'surface', 'volume',
        'points', 'lines', 'polygons', 'cells',
        'colors', 'texture', 'lighting', 'shading',
        'wireframe', 'solid', 'transparent',
        'camera view', 'perspective', 'orthographic',
        'chart', 'plot', 'graph', 'diagram'
    ]
    
    @staticmethod
    def extract_visual_concepts(example: Dict[str, Any]) -> List[str]:
        """Extract visual concepts from title, description, and category"""
        text = ' '.join([
            example.get('title', ''),
            example.get('explanation', ''),
            example.get('description', ''),
            example.get('category', '')
        ]).lower()
        
        concepts = []
        
        for concept in VisualConceptExtractor.VISUAL_CONCEPTS:
            if concept in text:
                concepts.append(concept)
        
        # Add category-based concepts
        category = example.get('category', '').lower()
        if 'geometric' in category:
            concepts.append('geometry')
        if 'volume' in category:
            concepts.append('volume rendering')
        if 'plot' in category or 'chart' in category:
            concepts.append('plotting')
        
        return sorted(list(set(concepts)))


class ImageChunker:
    """
    Creates metadata-only chunks for images
    
    Features:
    - No content (just metadata)
    - Bidirectional links to code + explanation chunks
    - Image type detection (result/diagram/baseline)
    - Visual concept extraction
    - Target: ~50 tokens (metadata only)
    """
    
    def __init__(self):
        self.type_detector = ImageTypeDetector()
        self.concept_extractor = VisualConceptExtractor()
    
    def chunk_image(
        self,
        example: Dict[str, Any],
        code_chunk_id: Optional[str] = None,
        explanation_chunk_id: Optional[str] = None
    ) -> Optional[ImageChunk]:
        """
        Create an image metadata chunk
        
        Args:
            example: Example dict with image info
            code_chunk_id: ID of related code chunk (if known)
            explanation_chunk_id: ID of related explanation chunk (if known)
        
        Returns:
            ImageChunk or None if no image
        """
        example_id = example.get('id', 'unknown')
        title = example.get('title', 'Untitled')
        
        # Check for image URL or local path
        image_url = example.get('image_url')
        local_image_path = example.get('local_image_path')
        
        if not image_url and not local_image_path:
            # No image for this example
            return None
        
        # Detect image type
        image_type = self.type_detector.detect_image_type(example)
        
        # Extract visual concepts
        visual_concepts = self.concept_extractor.extract_visual_concepts(example)
        
        # Build metadata
        metadata = {
            'title': title,
            'image_url': image_url,
            'local_path': local_image_path,
            'image_type': image_type,
            'shows_concepts': visual_concepts,
            'related_code_chunk': code_chunk_id or f"{title}Pythonic_code_0",
            'related_explanation_chunk': explanation_chunk_id or f"{title}_explanation_0"
        }
        
        # Build chunk ID
        chunk_id = f"{title}_image_0"
        
        # Content is empty for images (metadata-only)
        content = ""
        
        return ImageChunk(
            chunk_id=chunk_id,
            chunk_index=0,
            total_chunks=1,
            content=content,
            content_type="image",
            metadata=metadata,
            source_type=example.get('source_type', 'example'),
            original_id=example_id
        )


def main():
    """Test the image chunker"""
    print("=" * 80)
    print("ImageChunker - Test")
    print("=" * 80)
    
    # Test examples
    test_examples = [
        {
            'id': 'CylinderExample',
            'title': 'Cylinder Example',
            'explanation': 'This example demonstrates 3D rendering of a cylinder with solid shading.',
            'category': 'Geometric Objects',
            'image_url': 'https://vtk.org/examples/Cylinder.png',
            'local_image_path': 'data/images/Cylinder.png',
            'source_type': 'example'
        },
        {
            'id': 'VolumeRenderingBaseline',
            'title': 'Volume Rendering Baseline',
            'explanation': 'Expected output for volume rendering test.',
            'category': 'Volume Rendering',
            'image_url': 'https://vtk.org/baselines/VolumeRendering.png',
            'source_type': 'test'
        },
        {
            'id': 'NoImageExample',
            'title': 'No Image Example',
            'explanation': 'This example has no image.',
            'category': 'Data Processing',
            'source_type': 'example'
        }
    ]
    
    # Run chunker
    chunker = ImageChunker()
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n--- Test {i}: {example['title']} ---")
        
        chunk = chunker.chunk_image(example)
        
        if chunk:
            print(f"✓ Generated image chunk")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Tokens: ~{len(str(chunk.metadata)) // 4}")
            print(f"  Metadata:")
            print(f"    image_type: {chunk.metadata['image_type']}")
            print(f"    image_url: {chunk.metadata['image_url']}")
            print(f"    shows_concepts: {chunk.metadata['shows_concepts']}")
            print(f"    related_code_chunk: {chunk.metadata['related_code_chunk']}")
            print(f"    related_explanation_chunk: {chunk.metadata['related_explanation_chunk']}")
        else:
            print(f"✓ No image (correctly skipped)")
        
        print()
    
    print("=" * 80)
    print("✓ ImageChunker test complete")
    print("\nKey features demonstrated:")
    print("  • Metadata-only chunks (no content)")
    print("  • Image type detection (result/baseline)")
    print("  • Visual concept extraction")
    print("  • Bidirectional links to code + explanation")
    print("  • Skips examples without images")


if __name__ == '__main__':
    main()
