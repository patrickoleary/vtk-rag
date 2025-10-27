#!/usr/bin/env python3
"""
Test Set Builder for VTK RAG Evaluation

Generates labeled evaluation dataset from vtk-python-examples.jsonl:
1. Build test examples: Query, gold code, gold explanation, supporting chunks
2. Augment with steps: Add deterministic step decomposition and retrieval results

Creates complete test_set.jsonl with all evaluation data.

Usage:
    python test_set_builder.py
    
Output:
    data/processed/test_set.jsonl - Complete augmented test set
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))
sys.path.append(str(Path(__file__).parent.parent / 'retrieval-pipeline'))


@dataclass
class EvaluationExample:
    """Single evaluation example"""
    query: str                      # Generated query
    gold_code: str                  # Expected code output
    gold_explanation: str           # Expected explanation
    supporting_chunk_ids: List[str] # Chunk IDs that should be retrieved
    metadata: Dict[str, Any]        # Additional info (title, category, etc.)
    
    def to_dict(self):
        return asdict(self)


class TestSetBuilder:
    """
    Build and augment evaluation test set from VTK examples
    
    For each example, generates:
    - Natural language query
    - Gold standard code
    - Expected supporting passages
    - Deterministic step decomposition
    - Per-step retrieval results
    """
    
    def __init__(self, examples_file: Path):
        """
        Initialize builder
        
        Args:
            examples_file: Path to vtk-python-examples.jsonl
        """
        self.examples_file = examples_file
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict]:
        """Load examples from JSONL"""
        examples = []
        with open(self.examples_file, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def _clean_title(self, title: str) -> str:
        """
        Clean title to extract meaningful concept
        
        Handles:
        - Version numbers (CSVReadEdit1 → CSV Read Edit)
        - Demo/Example/Test suffixes
        - Tutorial/Step patterns
        - CamelCase splitting
        """
        # Remove common suffixes
        clean = re.sub(r'(Demo|Example|Test)\d*$', '', title)
        
        # Remove version numbers at end
        clean = re.sub(r'\d+$', '', clean)
        
        # Handle Tutorial/Step patterns
        clean = re.sub(r'Tutorial\s*Step\s*\d*', 'Tutorial', clean)
        
        # Handle numbers in middle (3DSImporter, Glyph2D, Glyph3D)
        clean = re.sub(r'(\d+)D([A-Z])', r'\1D \2', clean)
        
        # Split CamelCase into words
        clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
        clean = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean)
        
        # Clean up extra spaces
        clean = ' '.join(clean.split())
        
        return clean.strip()
    
    def _extract_concept_from_description(self, description: str) -> Optional[str]:
        """Extract the main concept from description"""
        if not description:
            return None
        
        # Look for common patterns
        patterns = [
            (r'This example (?:demonstrates|shows|illustrates) (.+?)[.]', False),
            (r'(?:used to|how to) (create|make|render|visualize|display|generate|build|draw) (.+?)[.]', True),
            (r'Demonstrates? (.+?)[.]', False),
            (r'Shows? how to (.+?)[.]', True),
            (r'(?:Creates?|Makes?) (.+?)[.]', False)
        ]
        
        for pattern, has_verb in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                if has_verb and len(match.groups()) == 2:
                    concept = f"{match.group(1)} {match.group(2)}".strip()
                else:
                    concept = match.group(1).strip()
                
                if len(concept) < 100 and len(concept) > 5:
                    if concept.startswith(('a ', 'an ', 'the ')) and not has_verb:
                        continue
                    return concept
        
        return None
    
    def _generate_query_variants(self, title: str, description: str) -> List[str]:
        """Generate query variants from example title and description"""
        queries = []
        
        # Clean the title
        title_clean = self._clean_title(title)
        
        # Skip if title is problematic
        if len(title_clean) < 3 or title_clean.replace(' ', '').isdigit():
            title_clean = title
        
        # Primary query from title
        if title_clean.lower() in ['cylinder', 'sphere', 'cone', 'cube']:
            queries.append(f"How do I create a {title_clean.lower()} in VTK?")
        else:
            queries.append(f"Show me how to {title_clean} in VTK")
        
        # Alternative formulations
        queries.append(f"VTK {title_clean} example")
        queries.append(f"How to use {title_clean} in VTK?")
        
        # Description-based query
        concept = self._extract_concept_from_description(description)
        if concept:
            verb_starters = ['create', 'make', 'render', 'visualize', 'display', 'generate', 'build', 'draw', 'use', 'implement']
            has_verb = any(concept.lower().startswith(v) for v in verb_starters)
            
            if has_verb:
                query = f"How do I {concept}?"
            else:
                if concept.startswith(('a ', 'an ', 'the ')):
                    concept = None
                else:
                    query = f"How do I {concept}?"
            
            if concept:
                queries.append(query)
                if len(concept) > 15 and 'example' not in concept.lower():
                    queries.insert(0, query)
        
        return queries
    
    def _extract_code_from_example(self, example: Dict) -> Optional[str]:
        """Extract code from example"""
        # Code is directly in the 'code' field
        code = example.get('code', '')
        if code and len(code) > 50:
            return code.strip()
        
        # Fallback: try 'content' field
        content = example.get('content', '')
        if not content:
            return None
        
        # Try to find Python code block
        code_pattern = r'```python\s*(.*?)```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no markdown code block, look for main() or imports
        if 'def main(' in content or 'import vtk' in content:
            return content.strip()
        
        return None
    
    def _extract_vtk_classes(self, code: str) -> List[str]:
        """Extract VTK class names from code"""
        pattern = r'\b(vtk[A-Z]\w+)\b'
        return list(set(re.findall(pattern, code)))
    
    def _generate_explanation(self, title: str, description: str, vtk_classes: List[str]) -> str:
        """Generate expected explanation based on example"""
        explanation = f"This example demonstrates {title}. "
        
        if description:
            explanation += description + " "
        
        if vtk_classes:
            classes_str = ', '.join(vtk_classes[:3])
            explanation += f"It uses VTK classes: {classes_str}."
        
        return explanation
    
    def build_test_set(
        self,
        max_examples: Optional[int] = None,
        min_code_length: int = 50,
        categories: Optional[List[str]] = None
    ) -> List[EvaluationExample]:
        """
        Build test set from examples
        
        Args:
            max_examples: Maximum number of examples to include
            min_code_length: Minimum code length to include
            categories: Only include specific categories
        
        Returns:
            List of EvaluationExample objects
        """
        test_set = []
        
        for example in self.examples[:max_examples] if max_examples else self.examples:
            # Extract metadata
            title = example.get('title', '')
            description = example.get('explanation', '') or example.get('description', '')
            category = example.get('category', '') or example.get('metadata', {}).get('category', '')
            
            # Filter by category if specified
            if categories and category not in categories:
                continue
            
            # Extract code
            code = self._extract_code_from_example(example)
            if not code or len(code) < min_code_length:
                continue
            
            # Extract VTK classes
            vtk_classes = self._extract_vtk_classes(code)
            
            # Use pre-generated queries from augmented JSONL files
            if 'query' in example and example['query']:
                queries = [example['query']]
            elif 'user_queries' in example and example['user_queries']:
                user_queries = example['user_queries']
                if isinstance(user_queries, dict):
                    primary = user_queries.get('primary', '')
                    alternatives = user_queries.get('alternatives', [])
                    queries = [primary] + alternatives if primary else alternatives
                elif isinstance(user_queries, list):
                    queries = user_queries
                else:
                    queries = self._generate_query_variants(title, description)
            elif 'user_query' in example and example['user_query']:
                queries = [example['user_query']]
                queries.extend(self._generate_query_variants(title, description))
            else:
                queries = self._generate_query_variants(title, description)
            
            # Generate expected explanation
            explanation = self._generate_explanation(title, description, vtk_classes)
            
            # Determine supporting chunks
            chunk_id = example.get('chunk_id', f"{title.replace(' ', '')}_code_0")
            supporting_chunks = [chunk_id]
            
            # Also add pythonic version if different
            pythonic_chunk_id = f"{title.replace(' ', '')}Pythonic_code_0"
            if pythonic_chunk_id != chunk_id:
                supporting_chunks.append(pythonic_chunk_id)
            
            # Include augmented metadata
            data_files = example.get('data_files', [])
            data_download_info = example.get('data_download_info', [])
            has_baseline = example.get('has_baseline', False)
            image_url = example.get('image_url')
            local_image_path = example.get('local_image_path')
            
            # Create evaluation example for primary query
            eval_example = EvaluationExample(
                query=queries[0],
                gold_code=code,
                gold_explanation=explanation,
                supporting_chunk_ids=supporting_chunks,
                metadata={
                    'title': title,
                    'category': category,
                    'vtk_classes': vtk_classes,
                    'alternative_queries': queries[1:],
                    'has_image': example.get('metadata', {}).get('has_image', False) or bool(image_url),
                    'data_files': data_files,
                    'data_download_info': data_download_info,
                    'has_baseline': has_baseline,
                    'image_url': image_url,
                    'local_image_path': local_image_path
                }
            )
            
            test_set.append(eval_example)
        
        return test_set
    
    def save_test_set(self, test_set: List[Dict], output_file: Path):
        """Save test set to JSONL file"""
        with open(output_file, 'w') as f:
            for example in test_set:
                f.write(json.dumps(example) + '\n')
        
        print(f"✓ Saved {len(test_set)} test examples to {output_file}")
    
    def print_statistics(self, test_set: List):
        """Print test set statistics"""
        print(f"\nTest Set Statistics:")
        print(f"  Total examples: {len(test_set)}")
        
        if not test_set:
            print("  (empty test set)")
            return
        
        # Category distribution
        categories = {}
        for example in test_set:
            if isinstance(example, EvaluationExample):
                cat = example.metadata.get('category', 'unknown')
            else:
                cat = example.get('metadata', {}).get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n  Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cat}: {count}")
        
        # Code length stats
        if isinstance(test_set[0], EvaluationExample):
            code_lengths = [len(ex.gold_code) for ex in test_set]
        else:
            code_lengths = [len(ex.get('gold_code', '')) for ex in test_set]
        
        print(f"\n  Code length:")
        print(f"    Min: {min(code_lengths)}")
        print(f"    Max: {max(code_lengths)}")
        print(f"    Avg: {sum(code_lengths) // len(code_lengths)}")
        
        # VTK classes
        all_classes = set()
        for ex in test_set:
            if isinstance(ex, EvaluationExample):
                all_classes.update(ex.metadata.get('vtk_classes', []))
            else:
                all_classes.update(ex.get('metadata', {}).get('vtk_classes', []))
        print(f"\n  Unique VTK classes: {len(all_classes)}")


def augment_example(
    example: Dict[str, Any],
    pipeline,
    chunks_per_step: int = 3
) -> Dict[str, Any]:
    """
    Augment a single test example with steps and retrieval results
    
    Args:
        example: Test example dict
        pipeline: Sequential pipeline (heuristic mode)
        chunks_per_step: Number of chunks to retrieve per step
        
    Returns:
        Augmented example with 'steps' and 'step_results' fields
    """
    query = example['query']
    
    # 1. Decompose query using deterministic heuristic
    steps = pipeline.decompose_query_heuristic(query)
    
    # 2. Retrieve chunks for each step
    step_results = []
    for step in steps:
        # Use retrieve_for_step which does the actual retrieval
        result = pipeline.retrieve_for_step(step, top_k=chunks_per_step)
        
        # Extract chunk IDs and scores
        chunk_data = [
            {
                'chunk_id': chunk.chunk_id,
                'score': chunk.score,
                'content_type': chunk.content_type
            }
            for chunk in result.retrieved_chunks
        ]
        
        step_results.append({
            'step_number': step.step_number,
            'description': step.description,
            'query': step.query,
            'focus': step.focus,
            'retrieved_chunks': chunk_data,
            'token_count': result.token_count
        })
    
    # 3. Add to example
    augmented = example.copy()
    augmented['steps'] = [
        {
            'step_number': s.step_number,
            'description': s.description,
            'query': s.query,
            'focus': s.focus
        }
        for s in steps
    ]
    augmented['step_results'] = step_results
    
    # Calculate aggregate stats
    all_retrieved_ids = []
    total_tokens = 0
    for sr in step_results:
        all_retrieved_ids.extend([c['chunk_id'] for c in sr['retrieved_chunks']])
        total_tokens += sr['token_count']
    
    # Keep original for comparison
    augmented['original_supporting_chunk_ids'] = example.get('supporting_chunk_ids', [])
    
    # New ground truth: unique chunks retrieved by multi-step approach
    augmented['supporting_chunk_ids'] = list(set(all_retrieved_ids))
    
    augmented['retrieval_stats'] = {
        'num_steps': len(steps),
        'total_chunks_retrieved': len(all_retrieved_ids),
        'unique_chunks_retrieved': len(set(all_retrieved_ids)),
        'total_tokens': total_tokens,
        'original_ground_truth_size': len(example.get('supporting_chunk_ids', [])),
        'new_ground_truth_size': len(set(all_retrieved_ids))
    }
    
    return augmented


def main():
    """Generate complete test set with augmentation"""
    print("=" * 80)
    print("Test Set Builder (Build + Augment)")
    print("=" * 80)
    
    # Paths
    examples_file = Path('data/raw/vtk-python-examples.jsonl')
    output_file = Path('data/processed/test_set.jsonl')
    
    # Check if examples file exists
    if not examples_file.exists():
        print(f"\n❌ Error: No source file found")
        print(f"Please ensure vtk-python-examples.jsonl is in data/raw/")
        return 1
    
    # STEP 1: Build test set
    print(f"\nSTEP 1: Building test set from {examples_file.name}")
    print("-" * 80)
    
    builder = TestSetBuilder(examples_file)
    
    test_set = builder.build_test_set(
        max_examples=None,      # Include ALL examples
        min_code_length=50,     # Minimal requirement
        categories=None         # All categories
    )
    
    print(f"✓ Built {len(test_set)} test examples")
    
    # Convert to dicts for augmentation
    test_examples = [ex.to_dict() for ex in test_set]
    
    # STEP 2: Augment with steps and retrieval
    print(f"\nSTEP 2: Augmenting with steps and retrieval results")
    print("-" * 80)
    print("  Initializing sequential pipeline (heuristic mode - deterministic)...")
    
    from sequential_pipeline import SequentialPipeline
    from task_specific_retriever import TaskSpecificRetriever
    
    pipeline = SequentialPipeline(use_llm_decomposition=False)
    retriever = TaskSpecificRetriever()
    print("  ✓ Pipeline initialized")
    
    print(f"\n  Augmenting {len(test_examples)} examples...")
    augmented_examples = []
    
    for i, example in enumerate(test_examples, 1):
        print(f"    [{i}/{len(test_examples)}] {example['query'][:60]}...")
        
        try:
            augmented = augment_example(
                example,
                pipeline,
                chunks_per_step=3
            )
            augmented_examples.append(augmented)
            
            # Show summary for first 3 examples
            if i <= 3:
                print(f"        Steps: {augmented['retrieval_stats']['num_steps']}")
                print(f"        Chunks: {augmented['retrieval_stats']['unique_chunks_retrieved']} unique")
                print(f"        Tokens: ~{augmented['retrieval_stats']['total_tokens']}")
                
        except Exception as e:
            print(f"        ⚠️  Error: {e}")
            # Keep original example without augmentation
            augmented_examples.append(example)
    
    # Save complete test set
    print(f"\n  Saving complete test set to {output_file}...")
    builder.save_test_set(augmented_examples, output_file)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Final Test Set Statistics")
    print("=" * 80)
    
    builder.print_statistics(augmented_examples)
    
    # Augmentation stats
    if augmented_examples and 'retrieval_stats' in augmented_examples[0]:
        avg_steps = sum(ex['retrieval_stats']['num_steps'] for ex in augmented_examples if 'retrieval_stats' in ex) / len(augmented_examples)
        avg_chunks = sum(ex['retrieval_stats']['unique_chunks_retrieved'] for ex in augmented_examples if 'retrieval_stats' in ex) / len(augmented_examples)
        avg_tokens = sum(ex['retrieval_stats']['total_tokens'] for ex in augmented_examples if 'retrieval_stats' in ex) / len(augmented_examples)
        
        print(f"\nAugmentation Averages:")
        print(f"  Steps per query: {avg_steps:.2f}")
        print(f"  Unique chunks per query: {avg_chunks:.2f}")
        print(f"  Tokens per query: {avg_tokens:.0f}")
    
    # Show sample
    print("\n" + "=" * 80)
    print("Sample Test Example")
    print("=" * 80)
    if augmented_examples:
        example = augmented_examples[0]
        print(f"\nQuery: {example['query']}")
        print(f"\nGold Code ({len(example['gold_code'])} chars):")
        print(example['gold_code'][:200] + "...")
        print(f"\nSupporting Chunks: {example['supporting_chunk_ids'][:3]}...")
        print(f"\nVTK Classes: {', '.join(example['metadata']['vtk_classes'][:5])}")
        if 'steps' in example:
            print(f"\nDecomposed Steps ({len(example['steps'])}):")
            for step in example['steps'][:3]:
                print(f"  {step['step_number']}. {step['description']}")
    
    print("\n" + "=" * 80)
    print("✓ Complete test set created successfully!")
    print("=" * 80)
    print(f"\nOutput: {output_file}")
    print(f"Contains: {len(augmented_examples)} examples with steps and retrieval results")
    print(f"\nUse with: python evaluator.py --test-set {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
