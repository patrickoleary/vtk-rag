#!/usr/bin/env python3
"""
Corpus Chunking Orchestrator

Applies all chunkers to raw data and outputs to data/processed/
Part of Phase 7 of the redesign.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

from code_chunker import CodeOnlyChunker
from explanation_chunker import ExplanationChunker
from image_chunker import ImageChunker
from api_doc_chunker import ApiDocChunker
from query_generator import QueryGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CorpusChunker:
    """Orchestrates chunking of entire VTK corpus"""
    
    def __init__(self, raw_dir: Path, output_dir: Path):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        
        # Initialize chunkers
        self.code_chunker = CodeOnlyChunker()
        self.explanation_chunker = ExplanationChunker()
        self.image_chunker = ImageChunker()
        self.api_doc_chunker = ApiDocChunker()
        self.query_generator = QueryGenerator()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def chunk_all(self):
        """Chunk all raw data files"""
        logger.info("=" * 80)
        logger.info("VTK Corpus Chunking - Phase 7")
        logger.info("=" * 80)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare output files
        output_files = {
            'code': self.output_dir / 'code_chunks.jsonl',
            'explanation': self.output_dir / 'explanation_chunks.jsonl',
            'api_doc': self.output_dir / 'api_doc_chunks.jsonl',
            'image': self.output_dir / 'image_chunks.jsonl'
        }
        
        # Open all output files
        file_handles = {
            key: open(path, 'w') for key, path in output_files.items()
        }
        
        try:
            # Process examples
            examples_file = self.raw_dir / 'vtk-python-examples.jsonl'
            if examples_file.exists():
                logger.info(f"\nProcessing {examples_file}")
                self._process_examples(examples_file, file_handles)
            else:
                logger.warning(f"Not found: {examples_file}")
            
            # Process tests
            tests_file = self.raw_dir / 'vtk-python-tests.jsonl'
            if tests_file.exists():
                logger.info(f"\nProcessing {tests_file}")
                self._process_examples(tests_file, file_handles)
            else:
                logger.warning(f"Not found: {tests_file}")
            
            # Process API docs
            docs_file = self.raw_dir / 'vtk-python-docs.jsonl'
            if docs_file.exists():
                logger.info(f"\nProcessing {docs_file}")
                self._process_api_docs(docs_file, file_handles)
            else:
                logger.warning(f"Not found: {docs_file}")
        
        finally:
            # Close all files
            for fh in file_handles.values():
                fh.close()
        
        # Generate statistics
        self._write_statistics()
        
        # Print summary
        self._print_summary()
    
    def _process_examples(self, filepath: Path, file_handles: Dict):
        """Process examples or tests file"""
        logger.info(f"  Reading examples from {filepath.name}")
        
        doc_count = 0
        
        with open(filepath) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    doc_count += 1
                    
                    # Enhance with query generator
                    code_queries = doc.get('code_queries', doc.get('user_queries', []))
                    vtk_classes = self.code_chunker.vtk_analyzer.extract_vtk_classes(doc.get('code', ''))
                    
                    # Generate explanation queries
                    if code_queries:
                        explanation_queries = self.query_generator.generate_explanation_queries(
                            code_queries=code_queries,
                            vtk_classes=vtk_classes
                        )
                    else:
                        explanation_queries = []
                    
                    # Chunk code
                    code_chunks = self.code_chunker.chunk_example(doc)
                    for chunk in code_chunks:
                        file_handles['code'].write(json.dumps(chunk.to_dict()) + '\n')
                        self.stats['code_chunks'] += 1
                    
                    # Chunk explanation
                    explanation_chunks = self.explanation_chunker.chunk_example(doc, code_queries)
                    for chunk in explanation_chunks:
                        # Add generated explanation queries
                        chunk.metadata['explanation_queries'] = explanation_queries
                        file_handles['explanation'].write(json.dumps(chunk.to_dict()) + '\n')
                        self.stats['explanation_chunks'] += 1
                    
                    # Chunk image (if exists)
                    if doc.get('image_url') or doc.get('local_image_path'):
                        image_chunk = self.image_chunker.chunk_image(doc)
                        if image_chunk:
                            file_handles['image'].write(json.dumps(image_chunk.to_dict()) + '\n')
                            self.stats['image_chunks'] += 1
                    
                    # Progress logging
                    if doc_count % 100 == 0:
                        logger.info(f"  Processed {doc_count} documents...")
                
                except json.JSONDecodeError:
                    logger.warning(f"  Skipping malformed JSON line")
                except Exception as e:
                    logger.error(f"  Error processing document: {e}")
        
        logger.info(f"  ✓ Processed {doc_count} documents from {filepath.name}")
    
    def _process_api_docs(self, filepath: Path, file_handles: Dict):
        """Process API documentation file"""
        logger.info(f"  Reading API docs from {filepath.name}")
        
        doc_count = 0
        
        with open(filepath) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    doc_count += 1
                    
                    # Chunk API doc
                    api_chunks = self.api_doc_chunker.chunk_api_doc(doc)
                    for chunk in api_chunks:
                        file_handles['api_doc'].write(json.dumps(chunk.to_dict()) + '\n')
                        self.stats['api_doc_chunks'] += 1
                    
                    # Progress logging
                    if doc_count % 100 == 0:
                        logger.info(f"  Processed {doc_count} classes...")
                
                except json.JSONDecodeError:
                    logger.warning(f"  Skipping malformed JSON line")
                except Exception as e:
                    logger.error(f"  Error processing API doc: {e}")
        
        logger.info(f"  ✓ Processed {doc_count} API docs from {filepath.name}")
    
    def _write_statistics(self):
        """Write statistics to JSON file"""
        stats = {
            'total_chunks': sum(self.stats.values()),
            'by_type': dict(self.stats),
            'output_files': {
                'code_chunks': 'data/processed/code_chunks.jsonl',
                'explanation_chunks': 'data/processed/explanation_chunks.jsonl',
                'api_doc_chunks': 'data/processed/api_doc_chunks.jsonl',
                'image_chunks': 'data/processed/image_chunks.jsonl'
            },
            'chunk_type_distribution': {
                'code': self.stats['code_chunks'],
                'explanation': self.stats['explanation_chunks'],
                'api_doc': self.stats['api_doc_chunks'],
                'image': self.stats['image_chunks']
            }
        }
        
        stats_file = self.output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\n✓ Statistics written to {stats_file}")
    
    def _print_summary(self):
        """Print summary of chunking"""
        logger.info("\n" + "=" * 80)
        logger.info("CHUNKING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nTotal chunks generated: {sum(self.stats.values()):,}")
        logger.info(f"\nBreakdown by type:")
        logger.info(f"  CODE chunks:        {self.stats['code_chunks']:,}")
        logger.info(f"  EXPLANATION chunks: {self.stats['explanation_chunks']:,}")
        logger.info(f"  API_DOC chunks:     {self.stats['api_doc_chunks']:,}")
        logger.info(f"  IMAGE chunks:       {self.stats['image_chunks']:,}")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info(f"\nNext step: Phase 8 - Rebuild Qdrant index")


def main():
    """Main entry point"""
    # Paths (relative to project root)
    raw_dir = Path('data/raw')
    output_dir = Path('data/processed')
    
    # Verify raw data exists
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.error("Expected structure:")
        logger.error("  data/raw/vtk-python-examples.jsonl")
        logger.error("  data/raw/vtk-python-tests.jsonl")
        logger.error("  data/raw/vtk-python-docs.jsonl")
        return
    
    # Create orchestrator
    chunker = CorpusChunker(raw_dir, output_dir)
    
    # Run chunking
    chunker.chunk_all()


if __name__ == '__main__':
    main()
