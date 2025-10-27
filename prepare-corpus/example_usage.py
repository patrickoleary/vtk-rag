#!/usr/bin/env python3
"""
Example Usage of Chunked VTK Corpus

This script demonstrates how to work with the chunked corpus data,
which separates chunks by content type (code, explanation, api_doc, image).
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from a JSONL file"""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def filter_by_content_type(chunks: List[Dict], content_type: str) -> List[Dict]:
    """Filter chunks by content type (code, explanation, api_doc, image)"""
    return [c for c in chunks if c.get('content_type') == content_type]


def filter_by_source_type(chunks: List[Dict], source_type: str) -> List[Dict]:
    """Filter chunks by source type (example, test, api)"""
    return [c for c in chunks if c.get('source_type') == source_type]


def filter_pythonic_code(chunks: List[Dict]) -> List[Dict]:
    """Filter code chunks that use Pythonic style (modular imports)"""
    return [c for c in chunks 
            if c.get('content_type') == 'code'
            and c.get('metadata', {}).get('source_style') == 'pythonic']


def filter_by_vtk_classes(chunks: List[Dict], class_names: List[str]) -> List[Dict]:
    """Filter chunks that use any of the specified VTK classes"""
    filtered = []
    for chunk in chunks:
        chunk_classes = chunk.get('metadata', {}).get('vtk_classes', [])
        if any(cls in chunk_classes for cls in class_names):
            filtered.append(chunk)
    return filtered


def filter_self_contained(chunks: List[Dict]) -> List[Dict]:
    """Filter code chunks that don't require external data files"""
    return [c for c in chunks 
            if c.get('content_type') == 'code'
            and not c.get('metadata', {}).get('requires_data_files', False)]


def filter_with_visualization(chunks: List[Dict]) -> List[Dict]:
    """Filter code chunks that include visualization"""
    return [c for c in chunks 
            if c.get('content_type') == 'code'
            and c.get('metadata', {}).get('has_visualization', False)]


def filter_by_complexity(chunks: List[Dict], complexity: str) -> List[Dict]:
    """Filter code chunks by complexity (simple, moderate, complex)"""
    return [c for c in chunks 
            if c.get('content_type') == 'code'
            and c.get('metadata', {}).get('complexity') == complexity]


def group_by_original_doc(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by their original document"""
    grouped = defaultdict(list)
    for chunk in chunks:
        original_id = chunk.get('original_id', 'unknown')
        grouped[original_id].append(chunk)
    
    # Sort chunks within each group by index
    for original_id in grouped:
        grouped[original_id].sort(key=lambda c: c.get('chunk_index', 0))
    
    return dict(grouped)


def search_content(chunks: List[Dict], query: str, case_sensitive: bool = False) -> List[Dict]:
    """Simple text search in chunk content"""
    if not case_sensitive:
        query = query.lower()
    
    results = []
    for chunk in chunks:
        content = chunk.get('content', '')
        if not case_sensitive:
            content = content.lower()
        
        if query in content:
            results.append(chunk)
    
    return results


def print_chunk_summary(chunk: Dict):
    """Print a summary of a chunk"""
    print(f"\nChunk: {chunk.get('chunk_id')}")
    print(f"  Content Type: {chunk.get('content_type')}")
    print(f"  Source Type: {chunk.get('source_type')}")
    print(f"  Index: {chunk.get('chunk_index')}/{chunk.get('total_chunks', 1) - 1}")
    
    metadata = chunk.get('metadata', {})
    
    # Code chunk metadata
    if chunk.get('content_type') == 'code':
        print(f"  Style: {metadata.get('source_style', 'N/A')}")
        print(f"  Import Style: {metadata.get('import_style', 'N/A')}")
        print(f"  Complexity: {metadata.get('complexity', 'N/A')}")
        print(f"  Has Visualization: {metadata.get('has_visualization', False)}")
        print(f"  Requires Data Files: {metadata.get('requires_data_files', False)}")
        
        if metadata.get('vtk_classes'):
            print(f"  VTK Classes: {', '.join(metadata['vtk_classes'][:5])}")
        
        if metadata.get('data_files'):
            print(f"  Data Files: {', '.join(metadata['data_files'])}")
    
    # Explanation chunk metadata
    elif chunk.get('content_type') == 'explanation':
        if 'title' in metadata:
            print(f"  Title: {metadata['title']}")
        if 'category' in metadata:
            print(f"  Category: {metadata['category']}")
    
    # API doc chunk metadata
    elif chunk.get('content_type') == 'api_doc':
        if 'class_name' in metadata:
            print(f"  Class: {metadata['class_name']}")
        if 'module_name' in metadata:
            print(f"  Module: {metadata['module_name']}")
        if 'method_names' in metadata:
            print(f"  Methods: {', '.join(metadata['method_names'][:3])}")
    
    # Image chunk metadata
    elif chunk.get('content_type') == 'image':
        if 'image_url' in metadata:
            print(f"  Image URL: {metadata['image_url']}")
        if 'image_title' in metadata:
            print(f"  Title: {metadata['image_title']}")
    
    content_preview = chunk.get('content', '')[:200].replace('\n', ' ')
    print(f"  Preview: {content_preview}...")


# Example usage demonstrations
def main():
    # Chunks are in data/processed/
    data_dir = Path('data/processed')
    
    print("VTK RAG Corpus - Example Usage\n")
    print("=" * 80)
    
    # Example 1: Load CODE chunks
    print("\n1. Working with CODE Chunks")
    print("-" * 80)
    
    code_file = data_dir / 'code_chunks.jsonl'
    if code_file.exists():
        code_chunks = load_chunks(code_file)
        
        # Filter Pythonic code
        pythonic = filter_pythonic_code(code_chunks)
        
        # Filter self-contained examples (no data files)
        self_contained = filter_self_contained(code_chunks)
        
        # Filter simple examples
        simple = filter_by_complexity(code_chunks, 'simple')
        
        print(f"Total CODE chunks: {len(code_chunks)}")
        print(f"Pythonic style: {len(pythonic)}")
        print(f"Self-contained (no data files): {len(self_contained)}")
        print(f"Simple complexity: {len(simple)}")
        
        if pythonic:
            print("\nExample Pythonic code chunk:")
            print_chunk_summary(pythonic[0])
    
    # Example 2: Load EXPLANATION chunks
    print("\n\n2. Working with EXPLANATION Chunks")
    print("-" * 80)
    
    explanation_file = data_dir / 'explanation_chunks.jsonl'
    if explanation_file.exists():
        explanation_chunks = load_chunks(explanation_file)
        
        print(f"Total EXPLANATION chunks: {len(explanation_chunks)}")
        
        if explanation_chunks:
            print("\nExample explanation chunk:")
            print_chunk_summary(explanation_chunks[0])
    
    # Example 3: Load API_DOC chunks
    print("\n\n3. Working with API_DOC Chunks")
    print("-" * 80)
    
    api_file = data_dir / 'api_doc_chunks.jsonl'
    if api_file.exists():
        api_chunks = load_chunks(api_file)
        
        # Search for specific class
        actor_docs = [c for c in api_chunks 
                     if c.get('metadata', {}).get('class_name') == 'vtkActor']
        
        print(f"Total API_DOC chunks: {len(api_chunks)}")
        print(f"vtkActor documentation chunks: {len(actor_docs)}")
        
        if actor_docs:
            print("\nExample API doc chunk:")
            print_chunk_summary(actor_docs[0])
    
    # Example 4: Load IMAGE chunks
    print("\n\n4. Working with IMAGE Chunks")
    print("-" * 80)
    
    image_file = data_dir / 'image_chunks.jsonl'
    if image_file.exists():
        image_chunks = load_chunks(image_file)
        
        print(f"Total IMAGE chunks: {len(image_chunks)}")
        
        if image_chunks:
            print("\nExample image chunk:")
            print_chunk_summary(image_chunks[0])
    
    # Example 5: Search by VTK classes
    print("\n\n5. Searching by VTK Classes")
    print("-" * 80)
    
    if code_chunks:
        # Find chunks using specific VTK classes
        actor_chunks = filter_by_vtk_classes(code_chunks, ['vtkActor'])
        renderer_chunks = filter_by_vtk_classes(code_chunks, ['vtkRenderer'])
        
        print(f"Code chunks using vtkActor: {len(actor_chunks)}")
        print(f"Code chunks using vtkRenderer: {len(renderer_chunks)}")
    
    # Example 6: Content search across all chunks
    print("\n\n6. Content Search Example")
    print("-" * 80)
    
    all_chunks = []
    for file_path in data_dir.glob('*_chunks.jsonl'):
        all_chunks.extend(load_chunks(file_path))
    
    if all_chunks:
        # Search for specific terms
        cylinder_chunks = search_content(all_chunks, 'cylinder')
        
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Chunks mentioning 'cylinder': {len(cylinder_chunks)}")
        
        # Group by content type
        by_type = defaultdict(int)
        for chunk in cylinder_chunks:
            by_type[chunk['content_type']] += 1
        
        print("\nBreakdown by content type:")
        for ctype, count in by_type.items():
            print(f"  {ctype}: {count}")
    
    print("\n\n" + "=" * 80)
    print("Example usage complete!")
    print("\nKey Features:")
    print("1. Chunks separated by content_type (code/explanation/api_doc/image)")
    print("2. Rich metadata for filtering (source_style, complexity, requires_data_files)")
    print("3. Support for Pythonic examples with modular imports")
    print("4. Separate image chunks with metadata links")
    print("\nFor RAG integration:")
    print("1. Use content_type filters for targeted retrieval")
    print("2. Prefer 'pythonic' source_style for generated code")
    print("3. Filter by 'requires_data_files' for self-contained examples")
    print("4. Link related chunks using related_*_chunk metadata")


if __name__ == '__main__':
    main()
