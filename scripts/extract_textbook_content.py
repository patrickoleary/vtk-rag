#!/usr/bin/env python3
"""
Extract content from vtk-python-textbook repository

Focuses on:
- Appendix B: Pythonic API documentation
- Markdown educational content
- Python examples demonstrating pythonic style

Output: vtk-python-textbook.jsonl for ingestion
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

class TextbookExtractor:
    """Extract and structure content from VTK Python textbook"""
    
    def __init__(self, textbook_repo_path: Path):
        """
        Initialize extractor
        
        Args:
            textbook_repo_path: Path to cloned vtk-python-textbook repo
        """
        self.textbook_path = textbook_repo_path
        self.appendix_b_path = textbook_repo_path / "appendix-b"
    
    def extract_all(self) -> List[Dict[str, Any]]:
        """Extract all relevant content"""
        documents = []
        
        # Extract Appendix B (Pythonic API)
        if self.appendix_b_path.exists():
            print(f"Processing Appendix B: {self.appendix_b_path}")
            documents.extend(self._extract_appendix_b())
        else:
            print(f"Warning: Appendix B not found at {self.appendix_b_path}")
        
        # Extract other markdown chapters
        documents.extend(self._extract_markdown_chapters())
        
        # Extract example notebooks if they exist
        documents.extend(self._extract_notebooks())
        
        return documents
    
    def _extract_appendix_b(self) -> List[Dict[str, Any]]:
        """Extract Appendix B content (Pythonic API documentation)"""
        documents = []
        
        # Look for markdown files in appendix-b
        for md_file in self.appendix_b_path.glob("*.md"):
            print(f"  Processing: {md_file.name}")
            
            content = md_file.read_text(encoding='utf-8')
            
            # Extract sections
            sections = self._split_markdown_sections(content)
            
            for i, section in enumerate(sections):
                doc = {
                    'doc_id': f"textbook_appendix_b_{md_file.stem}_{i}",
                    'source': 'vtk-python-textbook',
                    'chapter': 'Appendix B',
                    'file': md_file.name,
                    'title': section['title'],
                    'content': section['content'],
                    'type': 'educational',
                    'category': 'pythonic_api',
                    'has_code': '```python' in section['content'],
                    'topics': self._extract_topics(section['content'])
                }
                documents.append(doc)
        
        # Look for Python examples in appendix-b
        for py_file in self.appendix_b_path.glob("*.py"):
            print(f"  Processing example: {py_file.name}")
            
            code = py_file.read_text(encoding='utf-8')
            
            doc = {
                'doc_id': f"textbook_appendix_b_example_{py_file.stem}",
                'source': 'vtk-python-textbook',
                'chapter': 'Appendix B',
                'file': py_file.name,
                'title': f"Pythonic API Example: {py_file.stem}",
                'content': code,
                'type': 'code_example',
                'category': 'pythonic_api',
                'has_code': True,
                'import_style': 'pythonic' if 'vtkmodules' in code else 'monolithic'
            }
            documents.append(doc)
        
        return documents
    
    def _extract_markdown_chapters(self) -> List[Dict[str, Any]]:
        """Extract other educational markdown chapters"""
        documents = []
        
        # Look for markdown files in root and chapters/
        for md_file in self.textbook_path.glob("**/*.md"):
            # Skip appendix-b (already processed) and README
            if 'appendix-b' in str(md_file) or md_file.name.lower() == 'readme.md':
                continue
            
            print(f"  Processing chapter: {md_file.relative_to(self.textbook_path)}")
            
            content = md_file.read_text(encoding='utf-8')
            sections = self._split_markdown_sections(content)
            
            for i, section in enumerate(sections):
                doc = {
                    'doc_id': f"textbook_chapter_{md_file.stem}_{i}",
                    'source': 'vtk-python-textbook',
                    'chapter': md_file.parent.name if md_file.parent != self.textbook_path else 'main',
                    'file': md_file.name,
                    'title': section['title'],
                    'content': section['content'],
                    'type': 'educational',
                    'category': 'tutorial',
                    'has_code': '```python' in section['content'],
                    'topics': self._extract_topics(section['content'])
                }
                documents.append(doc)
        
        return documents
    
    def _extract_notebooks(self) -> List[Dict[str, Any]]:
        """Extract Jupyter notebooks if present"""
        documents = []
        
        # Look for .ipynb files
        for nb_file in self.textbook_path.glob("**/*.ipynb"):
            print(f"  Processing notebook: {nb_file.relative_to(self.textbook_path)}")
            
            # Basic extraction - could be enhanced with nbformat
            try:
                nb_content = json.loads(nb_file.read_text(encoding='utf-8'))
                
                # Extract markdown and code cells
                all_content = []
                for cell in nb_content.get('cells', []):
                    if cell['cell_type'] in ['markdown', 'code']:
                        all_content.append(''.join(cell.get('source', [])))
                
                if all_content:
                    doc = {
                        'doc_id': f"textbook_notebook_{nb_file.stem}",
                        'source': 'vtk-python-textbook',
                        'chapter': nb_file.parent.name,
                        'file': nb_file.name,
                        'title': f"Tutorial: {nb_file.stem}",
                        'content': '\n\n'.join(all_content),
                        'type': 'notebook',
                        'category': 'tutorial',
                        'has_code': True
                    }
                    documents.append(doc)
            except Exception as e:
                print(f"  Warning: Could not process notebook {nb_file.name}: {e}")
        
        return documents
    
    def _split_markdown_sections(self, content: str) -> List[Dict[str, str]]:
        """Split markdown content by headers"""
        sections = []
        
        # Split by ## headers (or # if no ## found)
        lines = content.split('\n')
        current_section = {'title': 'Introduction', 'content': []}
        
        for line in lines:
            # Check for header
            if line.startswith('## '):
                # Save previous section
                if current_section['content']:
                    sections.append({
                        'title': current_section['title'],
                        'content': '\n'.join(current_section['content']).strip()
                    })
                
                # Start new section
                current_section = {
                    'title': line.replace('## ', '').strip(),
                    'content': []
                }
            elif line.startswith('# ') and not sections:
                # Top-level header (only if first section)
                current_section['title'] = line.replace('# ', '').strip()
            else:
                current_section['content'].append(line)
        
        # Add last section
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content']).strip()
            })
        
        return sections
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        topics = []
        
        # Look for VTK-related keywords
        keywords = [
            'vtkmodules', 'pythonic', 'import', 'pipeline', 'filter',
            'renderer', 'actor', 'mapper', 'source', 'data', 'visualization'
        ]
        
        content_lower = content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                topics.append(keyword)
        
        return topics


def main():
    """Main extraction process"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_textbook_content.py <path_to_vtk_python_textbook_repo>")
        print("\nExample:")
        print("  python extract_textbook_content.py ~/repos/vtk-python-textbook")
        sys.exit(1)
    
    textbook_path = Path(sys.argv[1])
    
    if not textbook_path.exists():
        print(f"Error: Textbook repository not found at {textbook_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Extracting VTK Python Textbook Content")
    print("=" * 80)
    print(f"Source: {textbook_path}")
    print()
    
    extractor = TextbookExtractor(textbook_path)
    documents = extractor.extract_all()
    
    print(f"\n✓ Extracted {len(documents)} documents")
    
    # Write to JSONL
    output_file = Path('data/raw/vtk-python-textbook.jsonl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    
    print(f"✓ Written to {output_file}")
    
    # Print summary
    print("\nContent Summary:")
    by_type = {}
    by_category = {}
    for doc in documents:
        doc_type = doc['type']
        category = doc.get('category', 'unknown')
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
    
    print("\nBy type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")
    
    print("\nBy category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Review vtk-python-textbook.jsonl")
    print("2. Run: python prepare-corpus/chunk_corpus.py")
    print("3. Run: python build-indexes/build_qdrant_index.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
