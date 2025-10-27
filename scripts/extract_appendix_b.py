#!/usr/bin/env python3
"""
Extract ONLY Appendix B from vtk-python-textbook

Appendix B contains the pythonic API documentation - exactly what the LLM needs
to understand vtkmodules imports and modern VTK Python patterns.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def extract_appendix_b(textbook_path: Path) -> List[Dict[str, Any]]:
    """
    Extract Appendix B content (Pythonic API documentation)
    
    Args:
        textbook_path: Path to vtk-python-textbook repository
        
    Returns:
        List of documents ready for JSONL export
    """
    appendix_b_path = textbook_path / "appendix-b"
    
    if not appendix_b_path.exists():
        raise FileNotFoundError(f"Appendix B not found at {appendix_b_path}")
    
    documents = []
    
    print(f"Processing Appendix B: {appendix_b_path}")
    
    # Process markdown files
    for md_file in sorted(appendix_b_path.glob("*.md")):
        print(f"  • {md_file.name}")
        
        content = md_file.read_text(encoding='utf-8')
        
        doc = {
            'doc_id': f"appendix_b_{md_file.stem}",
            'source': 'vtk-python-textbook',
            'chapter': 'Appendix B',
            'file': md_file.name,
            'title': f"Pythonic API: {md_file.stem.replace('-', ' ').title()}",
            'content': content,
            'type': 'pythonic_api_doc',
            'category': 'pythonic_api',
            'has_code': '```python' in content or '```' in content,
            'priority': 'high'  # High priority for retrieval
        }
        documents.append(doc)
    
    # Process Python example files
    for py_file in sorted(appendix_b_path.glob("*.py")):
        print(f"  • {py_file.name}")
        
        code = py_file.read_text(encoding='utf-8')
        
        doc = {
            'doc_id': f"appendix_b_example_{py_file.stem}",
            'source': 'vtk-python-textbook',
            'chapter': 'Appendix B',
            'file': py_file.name,
            'title': f"Pythonic API Example: {py_file.stem.replace('_', ' ').title()}",
            'content': code,
            'type': 'pythonic_api_example',
            'category': 'pythonic_api',
            'has_code': True,
            'import_style': 'pythonic',
            'priority': 'high'
        }
        documents.append(doc)
    
    return documents


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_appendix_b.py <path_to_vtk_python_textbook_repo>")
        print("\nExample:")
        print("  python extract_appendix_b.py ~/repos/vtk-python-textbook")
        print("\nThis will extract ONLY Appendix B (pythonic API docs)")
        sys.exit(1)
    
    textbook_path = Path(sys.argv[1]).expanduser()
    
    if not textbook_path.exists():
        print(f"Error: Textbook repository not found at {textbook_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Extracting Appendix B: Pythonic API Documentation")
    print("=" * 80)
    print(f"Source: {textbook_path}")
    print()
    
    try:
        documents = extract_appendix_b(textbook_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n✓ Extracted {len(documents)} documents from Appendix B")
    
    # Write to JSONL
    output_file = Path('data/raw/vtk-python-textbook-appendix-b.jsonl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    
    print(f"✓ Written to {output_file}")
    
    # Print summary
    print("\nContent Summary:")
    md_count = sum(1 for d in documents if d['type'] == 'pythonic_api_doc')
    py_count = sum(1 for d in documents if d['type'] == 'pythonic_api_example')
    
    print(f"  Markdown docs: {md_count}")
    print(f"  Python examples: {py_count}")
    print(f"  Total: {len(documents)}")
    
    print("\nFiles extracted:")
    for doc in documents:
        print(f"  • {doc['file']}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Review: data/raw/vtk-python-textbook-appendix-b.jsonl")
    print("2. Update: prepare-corpus/chunk_corpus.py (add textbook processing)")
    print("3. Rerun: python prepare-corpus/chunk_corpus.py")
    print("4. Rebuild: python build-indexes/build_qdrant_index.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
