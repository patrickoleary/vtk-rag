#!/usr/bin/env python3
"""
Complete RAG Pipeline Evaluator

Runs end-to-end evaluation:
1. Load test set
2. Run retrieval for each query
3. Run complete RAG pipeline
4. Measure retrieval metrics (Recall, nDCG, MRR)
5. Measure end-to-end metrics (Exactness, Completeness, Attributable)
6. Generate evaluation report

Usage:
    python evaluator.py --test-set test_set.jsonl --num-examples 10
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add pipeline modules to path
sys.path.append(str(Path(__file__).parent.parent / 'retrieval-pipeline'))
sys.path.append(str(Path(__file__).parent.parent / 'grounding-prompting'))
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))
sys.path.append(str(Path(__file__).parent.parent / 'post-processing'))

from retrieval_metrics import RetrievalEvaluator, RetrievalMetrics
from end_to_end_metrics import EndToEndEvaluator, EndToEndMetrics


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    retrieval_metrics: Dict[str, float]
    end_to_end_metrics: Dict[str, float]
    api_validation_metrics: Dict[str, Any]
    num_examples: int
    per_example_results: List[Dict[str, Any]]


class RAGEvaluator:
    """
    Complete RAG pipeline evaluator
    
    Evaluates retrieval quality and end-to-end answer quality
    """
    
    def __init__(self, enable_llm: bool = False, enable_validation: bool = True, enable_visual_testing: bool = False):
        """
        Initialize evaluator
        
        Args:
            enable_llm: If True, runs full pipeline with LLM (costs money)
                       If False, evaluates retrieval only
            enable_validation: If True, validates generated code for errors
            enable_visual_testing: If True, executes code in Docker and validates visual output
        """
        self.retrieval_evaluator = RetrievalEvaluator()
        self.e2e_evaluator = EndToEndEvaluator()
        self.enable_llm = enable_llm
        self.enable_validation = enable_validation
        self.enable_visual_testing = enable_visual_testing
        
        # Initialize pipeline components
        # Always initialize for retrieval testing (uses heuristic decomposition)
        self._init_pipeline()
        
        # Initialize visual evaluator (optional)
        if self.enable_visual_testing:
            self._init_visual_evaluator()
    
    def _init_pipeline(self):
        """Initialize RAG pipeline components"""
        try:
            from sequential_pipeline import SequentialPipeline
            from json_response_processor import JSONResponseProcessor
            
            # Sequential pipeline handles: decomposition → retrieval → generation
            # For retrieval testing: use heuristic (deterministic, free)
            # For e2e testing: use LLM (production system)
            self.sequential = SequentialPipeline(
                use_llm_decomposition=self.enable_llm,  # LLM only for e2e
                enable_validation=self.enable_validation,
                validation_max_retries=2
            )
            
            # JSON processor for response analysis
            self.json_processor = JSONResponseProcessor()
            
            # Sequential pipeline initialized
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.enable_llm = False
    
    def _init_visual_evaluator(self):
        """Initialize visual testing evaluator (optional)"""
        try:
            from visual_evaluator import VisualEvaluator
            
            # Check if visual testing is enabled via environment
            import os
            visual_tests_enabled = os.getenv('RUN_VISUAL_TESTS', '0') == '1'
            create_baselines = os.getenv('CREATE_BASELINES', '0') == '1'
            
            if not visual_tests_enabled:
                print("⚠️  Visual testing disabled (set RUN_VISUAL_TESTS=1 to enable)")
                self.enable_visual_testing = False
                self.visual_evaluator = None
                return
            
            self.visual_evaluator = VisualEvaluator(
                enable_execution=True,
                baseline_dir=Path(__file__).parent.parent / "tests" / "visual_testing" / "baselines",
                create_baselines=create_baselines,
                memory_limit="512m",
                timeout=30,
                similarity_threshold=0.95
            )
            
            print(f"✓ Visual evaluator initialized (baselines={'creating' if create_baselines else 'comparing'})")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize visual evaluator: {e}")
            import traceback
            traceback.print_exc()
            self.enable_visual_testing = False
            self.visual_evaluator = None
    
    def load_test_set(self, test_set_file: Path) -> List[Dict]:
        """Load test set from JSONL"""
        test_set = []
        with open(test_set_file, 'r') as f:
            for line in f:
                if line.strip():
                    test_set.append(json.loads(line))
        return test_set
    
    def evaluate_retrieval_only(
        self,
        test_set: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Evaluate retrieval quality using pre-computed step results
        
        Evaluates both per-query (aggregate) and per-step metrics.
        Uses pre-computed steps from test_set.jsonl (created by test_set_builder.py)
        for apples-to-apples comparison across different retrieval methods.
        
        Args:
            test_set: List of test examples (with step_results from test_set_builder.py)
            top_k: Number of results to use (applied to step_results)
        
        Returns:
            List of retrieval results with per-query and per-step metrics
        """
        results = []
        
        print(f"\nEvaluating retrieval on {len(test_set)} examples...")
        
        for i, example in enumerate(test_set, 1):
            print(f"  [{i}/{len(test_set)}] {example['query'][:50]}...")
            
            # Extract retrieved chunk IDs from step_results
            if 'step_results' not in example:
                print(f"    ⚠️  No step_results, skipping")
                continue
            
            relevant_ids = set(example['supporting_chunk_ids'])
            
            # Evaluate each step independently
            step_evaluations = []
            all_retrieved_ids = []
            
            for step_result in example['step_results']:
                step_chunks = step_result['retrieved_chunks'][:top_k]
                step_chunk_ids = [c['chunk_id'] for c in step_chunks]
                
                # Calculate metrics for this step
                step_metrics = self.retrieval_evaluator.evaluate_query(
                    step_chunk_ids,
                    relevant_ids
                )
                
                # Find which relevant chunks this step retrieved
                relevant_in_step = [cid for cid in step_chunk_ids if cid in relevant_ids]
                
                step_evaluations.append({
                    'step_number': step_result['step_number'],
                    'description': step_result['description'],
                    'focus': step_result['focus'],
                    'retrieved_chunk_ids': step_chunk_ids,
                    'relevant_chunks_found': relevant_in_step,
                    'num_relevant_found': len(relevant_in_step),
                    'metrics': step_metrics,
                    'token_count': step_result['token_count']
                })
                
                all_retrieved_ids.extend(step_chunk_ids)
            
            # Remove duplicates for aggregate query-level metrics
            seen = set()
            unique_retrieved_ids = []
            for chunk_id in all_retrieved_ids:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    unique_retrieved_ids.append(chunk_id)
            
            # Calculate aggregate query-level metrics
            query_metrics = self.retrieval_evaluator.evaluate_query(
                unique_retrieved_ids,
                relevant_ids
            )
            
            # Determine which steps found relevant chunks
            steps_with_relevant = [s for s in step_evaluations if s['num_relevant_found'] > 0]
            
            results.append({
                'query': example['query'],
                'retrieved_ids': unique_retrieved_ids,
                'relevant_ids': list(relevant_ids),
                'num_steps': len(example.get('steps', [])),
                'metrics': query_metrics,
                'step_evaluations': step_evaluations,
                'steps_with_relevant_chunks': len(steps_with_relevant),
                'total_relevant_found': len([cid for cid in unique_retrieved_ids if cid in relevant_ids])
            })
            
            # Track step stats silently
        
        return results
    
    def evaluate_end_to_end(
        self,
        test_set: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Evaluate complete pipeline with LLM generation
        
        Uses pre-computed retrieval results from step_results,
        then generates responses with LLM.
        
        Args:
            test_set: List of test examples (with step_results)
            top_k: Number of chunks to use for generation (per step)
        
        Returns:
            List of complete evaluation results
        """
        if not self.enable_llm:
            print("⚠️  LLM evaluation disabled")
            return []
        
        results = []
        
        print(f"\nRunning end-to-end evaluation on {len(test_set)} examples (calls LLM)...")
        
        for i, example in enumerate(test_set, 1):
            print(f"\n  [{i}/{len(test_set)}] {example['query'][:50]}...")
            
            try:
                # Run the unified query pipeline (returns JSON)
                response = self.sequential.process_query(example['query'])
                
                # Extract results from JSON response
                generated_code = response.get('code', '')
                generated_explanation = response.get('explanation', '')
                citations = response.get('citations', [])
                
                # For compatibility, extract chunk IDs from citations
                retrieved_ids = []
                if isinstance(citations, list):
                    for cite in citations:
                        if isinstance(cite, dict):
                            retrieved_ids.append(cite.get('number', 0))
                        else:
                            retrieved_ids.append(cite)
                
                # Process response for metadata
                processed = self.json_processor.process(response)
                
                if not generated_code:
                    print(f"    ⚠️  No code generated")
                
                # Extract augmented metadata from test example
                metadata_dict = example.get('metadata', {})
                expected_data_files = metadata_dict.get('data_files', [])
                # Check if ANY result image exists (image_url OR baseline_image)
                expected_has_image = bool(metadata_dict.get('image_url') or metadata_dict.get('has_baseline', False))
                
                # Create a mock parsed response for e2e evaluator
                # (evaluator expects old format with various attributes/methods)
                from types import SimpleNamespace
                
                # Convert citation dicts to objects with chunk_id attribute
                citation_objects = []
                for cite in citations:
                    if isinstance(cite, dict):
                        citation_objects.append(SimpleNamespace(
                            chunk_id=cite.get('number', 0),
                            reason=cite.get('reason', '')
                        ))
                    else:
                        citation_objects.append(cite)
                
                # Extract VTK classes from code
                import re
                vtk_classes = []
                if generated_code:
                    vtk_classes = list(set(re.findall(r'\bvtk[A-Z]\w+', generated_code)))
                
                # Create mock with all required attributes and methods
                class MockParsedResponse:
                    def __init__(self):
                        self.citations = citation_objects
                        self.has_citations = len(citation_objects) > 0
                        self.code_blocks = [generated_code] if generated_code else []
                        self.explanations = [generated_explanation] if generated_explanation else []
                        self.has_code = bool(generated_code)
                        self.raw_text = generated_code + "\n\n" + generated_explanation
                        self.data_files = []  # Extract from response if needed
                    
                    def get_all_vtk_classes(self):
                        return vtk_classes
                    
                    def format_data_section(self):
                        return ""
                    
                    def has_baseline_images(self):
                        return False
                
                mock_parsed = MockParsedResponse()
                
                # Calculate end-to-end metrics
                e2e_metrics = self.e2e_evaluator.evaluate_response(
                    generated_code=generated_code,
                    gold_code=example['gold_code'],
                    generated_explanation=generated_explanation,
                    parsed_response=mock_parsed,  # Use mock for compatibility
                    retrieved_chunk_ids=set(retrieved_ids) if retrieved_ids else set(),
                    relevant_chunk_ids=set(example['supporting_chunk_ids']),
                    expected_data_files=expected_data_files,
                    expected_has_image=expected_has_image
                )
                
                # Extract decomposition steps from response if available
                decomposition_steps = []
                if 'steps' in response:
                    for step in response['steps']:
                        decomposition_steps.append({
                            'step_number': step.get('step_number', 0),
                            'description': step.get('description', ''),
                            'query': step.get('query', step.get('search_query', '')),
                            'focus': step.get('focus', '')
                        })
                
                # Get validation info from response metadata if available
                validation_attempted = response.get('validation_attempted', False)
                validation_errors_found = response.get('validation_errors_found', 0)
                validation_retries = response.get('validation_retries', 0)
                validation_final_status = response.get('validation_final_status', 'not_run')
                
                # Get security validation info
                security_check_passed = response.get('security_check_passed', True)
                security_issues = response.get('security_issues', [])
                
                # Get API validation info (NEW)
                api_validation_attempted = response.get('api_validation_attempted', False)
                api_validation_passed = response.get('api_validation_passed', True)
                api_validation_errors = response.get('api_validation_errors', [])
                
                # NEW: Visual validation (optional)
                visual_metrics = {}
                if self.enable_visual_testing and self.visual_evaluator:
                    visual_metrics = self.visual_evaluator.evaluate_code(
                        code=generated_code,
                        test_name=f"query_{i}",
                        expected_has_output=expected_has_image
                    )
                    
                    if visual_metrics['execution_success']:
                        print(f"    ✓ Code executed successfully")
                    else:
                        print(f"    ✗ Execution failed: {visual_metrics['execution_error']}")
                    
                    if visual_metrics['has_visual_output']:
                        if visual_metrics['has_baseline']:
                            if visual_metrics['visual_regression_detected']:
                                print(f"    ⚠️  Visual regression (similarity={visual_metrics['visual_similarity']:.3f})")
                            else:
                                print(f"    ✓ Visual test passed (similarity={visual_metrics['visual_similarity']:.3f})")
                        else:
                            print(f"    ⓘ  No baseline for comparison")
                else:
                    # Visual testing disabled - set default values
                    visual_metrics = {
                        'execution_attempted': False,
                        'execution_success': None,
                        'visual_validation_passed': None
                    }
                
                results.append({
                    'query': example['query'],
                    'retrieved_ids': retrieved_ids,  # For debugging/analysis only
                    'relevant_ids': list(example['supporting_chunk_ids']),  # For report
                    'generated_code': generated_code,
                    'gold_code': example['gold_code'],
                    'generated_explanation': generated_explanation,
                    'parsed_response': mock_parsed,  # Use mock for compatibility
                    'expected_data_files': expected_data_files,
                    'expected_has_image': expected_has_image,
                    # NOTE: No retrieval_metrics in e2e mode - retrieval is not what we're evaluating
                    'e2e_metrics': e2e_metrics,
                    # Validation metrics
                    'validation_attempted': validation_attempted,
                    'validation_errors_found': validation_errors_found,
                    'validation_retries': validation_retries,
                    'validation_final_status': validation_final_status,
                    # Security validation metrics (NEW)
                    'security_check_passed': security_check_passed,
                    'security_issues': security_issues,
                    # API validation metrics (NEW)
                    'api_validation_attempted': api_validation_attempted,
                    'api_validation_passed': api_validation_passed,
                    'api_validation_errors': api_validation_errors,
                    'api_validation_errors_count': len(api_validation_errors),
                    'api_validation_fixed': api_validation_attempted and api_validation_passed and len(api_validation_errors) > 0,
                    # CRITICAL: Save LLM decomposition for analysis
                    'decomposition_steps': decomposition_steps,
                    'num_steps': len(decomposition_steps),
                    # Visual validation metrics (NEW)
                    **visual_metrics
                })
                
                print(f"    ✓ {len(decomposition_steps)} steps → {len(retrieved_ids)} chunks"
                      f"  |  Exactness: {e2e_metrics['code_exactness']:.1%}, "
                      f"Completeness: {e2e_metrics['code_completeness']:.1%}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def generate_report(
        self,
        results: List[Dict],
        output_file: Path,
        mode: str = "retrieval"
    ):
        """
        Generate evaluation report
        
        Args:
            results: Evaluation results
            output_file: Where to save report
            mode: "retrieval" or "end_to_end"
        """
        report = {
            'mode': mode,
            'num_examples': len(results)
        }
        
        if mode == "retrieval":
            # Calculate average retrieval metrics (query-level)
            retrieval_results = [
                {'retrieved_ids': r['retrieved_ids'], 'relevant_ids': r['relevant_ids']}
                for r in results
            ]
            metrics = self.retrieval_evaluator.evaluate_test_set(retrieval_results)
            report['retrieval_metrics'] = metrics.to_dict()
            
            # Calculate overall step statistics (not per-step-number - those are meaningless)
            all_steps = []
            for r in results:
                all_steps.extend(r.get('step_evaluations', []))
            
            if all_steps:
                # Overall summary stats only
                report['step_summary'] = {
                    'total_steps_executed': len(all_steps),
                    'avg_steps_per_query': len(all_steps) / len(results),
                    'total_steps_with_relevant': sum(1 for s in all_steps if s['num_relevant_found'] > 0),
                    'pct_steps_with_relevant': sum(1 for s in all_steps if s['num_relevant_found'] > 0) / len(all_steps) * 100
                }
                
                # Note: We don't aggregate by step number (step 1, step 2, etc.)
                # because steps are semantically different across queries
            
            report['per_example_results'] = results  # Save individual results for analysis
            
        elif mode == "end-to-end":
            # Calculate end-to-end code quality metrics (no retrieval metrics)
            e2e_results = [
                {
                    'generated_code': r['generated_code'],
                    'gold_code': r['gold_code'],
                    'generated_explanation': r['generated_explanation'],
                    'parsed_response': r['parsed_response'],
                    'retrieved_chunk_ids': r['retrieved_ids'],
                    'relevant_chunk_ids': r['relevant_ids'],
                    'expected_data_files': r.get('expected_data_files', []),
                    'expected_has_image': r.get('expected_has_image', False),
                    # Validation metrics
                    'validation_attempted': r.get('validation_attempted', False),
                    'validation_errors_found': r.get('validation_errors_found', 0),
                    'validation_retries': r.get('validation_retries', 0),
                    'validation_final_status': r.get('validation_final_status', 'not_run'),
                    # Security validation metrics (NEW)
                    'security_check_passed': r.get('security_check_passed', True),
                    'security_issues': r.get('security_issues', [])
                }
                for r in results
            ]
            e2e_metrics = self.e2e_evaluator.evaluate_test_set(e2e_results)
            report['end_to_end_metrics'] = e2e_metrics.to_dict()
            
            # Aggregate visual validation metrics (if any)
            visual_results = [r for r in results if r.get('execution_attempted')]
            if visual_results:
                total = len(visual_results)
                success_count = sum(1 for r in visual_results if r.get('execution_success'))
                has_output_count = sum(1 for r in visual_results if r.get('has_visual_output'))
                has_baseline_count = sum(1 for r in visual_results if r.get('has_baseline'))
                passed_regression_count = sum(
                    1 for r in visual_results 
                    if r.get('has_baseline') and not r.get('visual_regression_detected')
                )
                validation_passed_count = sum(1 for r in visual_results if r.get('visual_validation_passed'))
                
                total_execution_time = sum(r.get('execution_time', 0) for r in visual_results)
                total_output_size = sum(r.get('visual_output_size', 0) for r in visual_results)
                similarities = [r.get('visual_similarity') for r in visual_results if r.get('visual_similarity') is not None]
                
                report['visual_metrics'] = {
                    'execution_attempted': total / len(results),
                    'execution_success': success_count / total if total > 0 else 0,
                    'execution_failed': (total - success_count) / total if total > 0 else 0,
                    'avg_execution_time': total_execution_time / total if total > 0 else 0,
                    'has_visual_output': has_output_count / total if total > 0 else 0,
                    'missing_visual_output': (total - has_output_count) / total if total > 0 else 0,
                    'avg_output_size_kb': (total_output_size / total / 1024) if total > 0 else 0,
                    'with_baseline': has_baseline_count / total if total > 0 else 0,
                    'regression_passed': passed_regression_count / has_baseline_count if has_baseline_count > 0 else 0,
                    'regression_failed': (has_baseline_count - passed_regression_count) / has_baseline_count if has_baseline_count > 0 else 0,
                    'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
                    'validation_passed': validation_passed_count / total if total > 0 else 0
                }
            
            # Aggregate API validation metrics (NEW)
            api_attempted_count = sum(1 for r in results if r.get('api_validation_attempted'))
            if api_attempted_count > 0:
                # Count queries with API errors detected
                api_errors_detected_count = sum(
                    1 for r in results 
                    if r.get('api_validation_attempted') and r.get('api_validation_errors_count', 0) > 0
                )
                
                # Count queries where API errors were successfully fixed
                api_errors_fixed_count = sum(
                    1 for r in results
                    if r.get('api_validation_fixed', False)
                )
                
                # Count queries where API errors remain unfixed
                api_errors_unfixed_count = sum(
                    1 for r in results
                    if r.get('api_validation_attempted') 
                    and r.get('api_validation_errors_count', 0) > 0
                    and not r.get('api_validation_passed', True)
                )
                
                # Total error instances across all queries
                total_api_errors = sum(r.get('api_validation_errors_count', 0) for r in results)
                
                # Collect all error types
                all_error_types = {}
                for r in results:
                    for error in r.get('api_validation_errors', []):
                        # Extract error type from error string
                        error_type = error.split(':')[0] if ':' in error else 'unknown'
                        all_error_types[error_type] = all_error_types.get(error_type, 0) + 1
                
                report['api_validation_metrics'] = {
                    'validation_attempted_count': api_attempted_count,
                    'validation_attempted_pct': api_attempted_count / len(results),
                    'errors_detected_count': api_errors_detected_count,
                    'errors_detected_pct': api_errors_detected_count / len(results),
                    'errors_fixed_count': api_errors_fixed_count,
                    'errors_fixed_pct': api_errors_fixed_count / api_errors_detected_count if api_errors_detected_count > 0 else 0,
                    'errors_unfixed_count': api_errors_unfixed_count,
                    'errors_unfixed_pct': api_errors_unfixed_count / api_errors_detected_count if api_errors_detected_count > 0 else 0,
                    'total_error_instances': total_api_errors,
                    'avg_errors_per_query': total_api_errors / len(results),
                    'error_types': all_error_types,
                    'queries_with_errors': [
                        {
                            'query': r['query'][:80],
                            'errors_count': r.get('api_validation_errors_count', 0),
                            'fixed': r.get('api_validation_fixed', False),
                            'errors': r.get('api_validation_errors', [])
                        }
                        for r in results 
                        if r.get('api_validation_errors_count', 0) > 0
                    ]
                }
            else:
                report['api_validation_metrics'] = {
                    'validation_attempted_count': 0,
                    'message': 'API validation was not enabled for this evaluation'
                }
            
            report['per_example_results'] = results  # Save individual results
        
        # Save report (exclude non-serializable objects)
        serializable_report = report.copy()
        if 'per_example_results' in serializable_report:
            # Remove parsed_response objects which can't be serialized
            # Convert EnrichedResponse to dict if present
            serializable_results = []
            for r in serializable_report['per_example_results']:
                r_copy = {k: v for k, v in r.items() if k != 'parsed_response'}
                # If parsed_response is EnrichedResponse, convert to dict
                if 'parsed_response' in r:
                    try:
                        if hasattr(r['parsed_response'], 'to_dict'):
                            r_copy['parsed_response'] = r['parsed_response'].to_dict()
                    except:
                        pass  # Skip if can't serialize
                serializable_results.append(r_copy)
            serializable_report['per_example_results'] = serializable_results
        
        with open(output_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"\n✓ Report saved to {output_file}")
        
        # Save summary to text file
        summary_file = output_file.parent / (output_file.stem + '_summary.txt')
        self.save_summary(report, summary_file)
        print(f"✓ Summary saved to {summary_file}")
    
    def save_summary(self, report, summary_file):
        """Save formatted summary to text file"""
        with open(summary_file, 'w') as f:
            self._write_report(report, f)
    
    def _write_report(self, report, file):
        """Write evaluation report to file"""
        def write(text=""):
            file.write(text + "\n")
        
        write("=" * 80)
        write("EVALUATION REPORT")
        write("=" * 80)
        write(f"\nMode: {report['mode']}")
        write(f"Examples: {report['num_examples']}")
        
        # Only show retrieval metrics for retrieval mode
        if report['mode'] == 'retrieval' and 'retrieval_metrics' in report:
            write("\n" + "-" * 80)
            write("RETRIEVAL METRICS (Query-Level Aggregate)")
            write("-" * 80)
            metrics = report['retrieval_metrics']
            write(f"Recall@1:  {metrics['recall@1']:.3f}")
            write(f"Recall@3:  {metrics['recall@3']:.3f}")
            write(f"Recall@5:  {metrics['recall@5']:.3f}")
            write(f"Recall@10: {metrics['recall@10']:.3f}")
            write()
            write(f"nDCG@3:    {metrics['ndcg@3']:.3f}")
            write(f"nDCG@5:    {metrics['ndcg@5']:.3f}")
            write(f"nDCG@10:   {metrics['ndcg@10']:.3f}")
            write()
            write(f"MRR:       {metrics['mrr']:.3f}")
            
            # Write detailed retrieval analysis
            if 'per_example_results' in report:
                self._write_retrieval_analysis(report, write)
        
        if 'step_summary' in report:
            write("\n" + "-" * 80)
            write("OVERALL STEP STATISTICS")
            write("-" * 80)
            summary = report['step_summary']
            write(f"Total steps executed: {summary['total_steps_executed']}")
            write(f"Avg steps per query:  {summary['avg_steps_per_query']:.2f}")
            write(f"Steps with relevant:  {summary['total_steps_with_relevant']} ({summary['pct_steps_with_relevant']:.1f}%)")
        
        # E2E metrics (only for end-to-end mode)
        if report['mode'] == 'end-to-end' and 'end_to_end_metrics' in report:
            write("\n" + "-" * 80)
            write("END-TO-END CODE QUALITY METRICS")
            write("-" * 80)
            metrics = report['end_to_end_metrics']
            
            write("\nCode Quality:")
            write(f"  Exactness:     {metrics['code_exactness']:.1%} (similarity to gold)")
            write(f"  Correctness:   {metrics['code_correctness']:.1%} (has all components)")
            write(f"  Syntax Valid:  {metrics['code_syntax_valid']:.1%} (no syntax errors)")
            write()
            write("Explanation:")
            write(f"  Quality:       {metrics['explanation_present']:.1%}")
            write()
            write("Syntax Validation Metrics:")
            write(f"  Attempted:     {metrics['validation_attempted']:.1%}")
            write(f"  Errors Found:  {metrics['validation_errors_found']:.1%}")
            write(f"  Needed Retry:  {metrics['validation_needed_retry']:.1%}")
            write(f"  Success Rate:  {metrics['validation_success_rate']:.1%}")
            write(f"  Avg Retries:   {metrics['validation_avg_retries']:.2f}")
            write()
            write("Security Validation Metrics:")
            write(f"  Checked:       {metrics.get('security_check_performed', 0.0):.1%} (% with security check)")
            write(f"  Passed:        {metrics.get('security_check_passed', 1.0):.1%} (% that passed)")
            write(f"  Issues Found:  {metrics.get('security_issues_found', 0.0):.1%} (% with security issues)")
        
        # Visual validation metrics (if enabled)
        if report['mode'] == 'end-to-end' and 'visual_metrics' in report:
            write("\n" + "-" * 80)
            write("VISUAL VALIDATION METRICS")
            write("-" * 80)
            vis = report['visual_metrics']
            write("\nExecution:")
            write(f"  Attempted:     {vis['execution_attempted']:.1%}")
            write(f"  Success:       {vis['execution_success']:.1%}")
            write(f"  Failed:        {vis['execution_failed']:.1%}")
            write(f"  Avg Time:      {vis['avg_execution_time']:.2f}s")
            write()
            write("Visual Output:")
            write(f"  Has Output:    {vis['has_visual_output']:.1%}")
            write(f"  Missing:       {vis['missing_visual_output']:.1%}")
            write(f"  Avg Size:      {vis['avg_output_size_kb']:.1f} KB")
            write()
            write("Visual Regression:")
            write(f"  With Baseline: {vis['with_baseline']:.1%}")
            write(f"  Passed:        {vis['regression_passed']:.1%}")
            write(f"  Failed:        {vis['regression_failed']:.1%}")
            write(f"  Avg Similarity:{vis['avg_similarity']:.3f}")
            write()
            write("Overall:")
            write(f"  Validation Passed: {vis['validation_passed']:.1%}")
        
        # API validation metrics (if enabled) (NEW)
        if report['mode'] == 'end-to-end' and 'api_validation_metrics' in report:
            api = report['api_validation_metrics']
            
            if api.get('validation_attempted_count', 0) > 0:
                write("\n" + "-" * 80)
                write("API VALIDATION METRICS (VTK API Hallucination Detection)")
                write("-" * 80)
                write("\nValidation Coverage:")
                write(f"  Queries Validated: {api['validation_attempted_count']}/{report['num_examples']} ({api['validation_attempted_pct']:.1%})")
                write()
                write("Error Detection:")
                write(f"  Queries with Errors:  {api['errors_detected_count']}/{report['num_examples']} ({api['errors_detected_pct']:.1%})")
                write(f"  Total Error Instances:{api['total_error_instances']}")
                write(f"  Avg Errors per Query: {api['avg_errors_per_query']:.2f}")
                write()
                write("Error Repair:")
                write(f"  Successfully Fixed:   {api['errors_fixed_count']}/{api['errors_detected_count']} ({api['errors_fixed_pct']:.1%})")
                write(f"  Unfixed After Retry:  {api['errors_unfixed_count']}/{api['errors_detected_count']} ({api['errors_unfixed_pct']:.1%})")
                write()
                
                if api.get('error_types'):
                    write("Error Types:")
                    for error_type, count in sorted(api['error_types'].items(), key=lambda x: -x[1]):
                        write(f"  {error_type:20s}: {count}")
                
                if api.get('queries_with_errors'):
                    write()
                    write("Queries with API Errors:")
                    for i, q in enumerate(api['queries_with_errors'], 1):
                        status = "✓ FIXED" if q['fixed'] else "✗ UNFIXED"
                        write(f"  {i}. [{status}] {q['query']}")
                        write(f"     Errors: {q['errors_count']}")
                        for err in q['errors'][:2]:  # Show first 2 errors
                            write(f"       - {err[:100]}")
    
    def _write_retrieval_analysis(self, report, write):
        """Write detailed retrieval analysis to file"""
        results = report['per_example_results']
        
        write("\n" + "-" * 80)
        write("DETAILED RETRIEVAL ANALYSIS")
        write("-" * 80)
        
        # Calculate Recall@N progression
        recall_values = {}
        for n in [1, 3, 5, 10, 15, 20]:
            total_recall = 0
            for ex in results:
                retrieved = ex['retrieved_ids'][:n]
                relevant = set(ex['relevant_ids'])
                found = len(set(retrieved) & relevant)
                total_relevant = len(relevant)
                recall_at_n = found / total_relevant if total_relevant > 0 else 0
                total_recall += recall_at_n
            recall_values[n] = total_recall / len(results)
        
        write("\nRecall@N Progression:")
        for n, recall in recall_values.items():
            write(f"  Recall@{n:2d}: {recall:.3f} ({recall*100:.1f}%)")
        
        # Analysis of ground truth size
        relevant_counts = [len(ex['relevant_ids']) for ex in results]
        avg_relevant = sum(relevant_counts) / len(relevant_counts)
        max_relevant = max(relevant_counts)
        
        write(f"\nGround Truth Statistics:")
        write(f"  Avg relevant chunks per query: {avg_relevant:.2f}")
        write(f"  Max relevant chunks: {max_relevant}")
        write(f"  Queries with >10 relevant: {sum(1 for c in relevant_counts if c > 10)} ({sum(1 for c in relevant_counts if c > 10)/len(relevant_counts)*100:.1f}%)")
        
        # Check for perfect recall
        perfect_recall_at_20 = sum(1 for ex in results 
                                    if len(set(ex['retrieved_ids'][:20]) & set(ex['relevant_ids'])) == len(ex['relevant_ids']))
        
        write(f"\nSystem Performance:")
        write(f"  Queries with 100% recall in top 20: {perfect_recall_at_20}/{len(results)} ({perfect_recall_at_20/len(results)*100:.1f}%)")
        
        if recall_values[20] >= 0.999:
            write(f"  ✓ System finds ALL relevant chunks (Recall@20 ≈ 100%)")
        
        if report['retrieval_metrics']['ndcg@10'] >= 0.95:
            write(f"  ✓ Excellent ranking quality (nDCG@10 = {report['retrieval_metrics']['ndcg@10']:.3f})")
        
        if report['retrieval_metrics']['recall@10'] < avg_relevant / 10:
            write(f"  ℹ Recall@10 limited by ground truth size (avg {avg_relevant:.1f} relevant > 10)")
        
        # Find examples with low performance
        low_recall_queries = []
        for ex in results:
            retrieved = set(ex['retrieved_ids'][:20])
            relevant = set(ex['relevant_ids'])
            recall = len(retrieved & relevant) / len(relevant) if relevant else 0
            if recall < 0.9:
                low_recall_queries.append((ex['query'], recall, len(relevant)))
        
        if low_recall_queries:
            write(f"\n  ⚠ {len(low_recall_queries)} queries with <90% recall in top 20:")
            for query, recall, total in low_recall_queries[:3]:
                write(f"    - {query[:60]}... ({recall:.1%}, {total} relevant)")
        else:
            write(f"  ✓ All queries achieve ≥90% recall in top 20")
        
        write("\n" + "=" * 80)
        write("CONCLUSION")
        write("=" * 80)
        if recall_values[20] >= 0.999 and report['retrieval_metrics']['ndcg@10'] >= 0.95:
            write("✓ Retrieval system performing EXCELLENTLY")
            write("  - Finds all relevant chunks")
            write("  - Ranks them well (most in top 10)")
            write("  - Production ready!")
        elif recall_values[10] >= 0.85:
            write("✓ Retrieval system performing WELL")
            write("  - Finds most relevant chunks in top 10")
            write("  - Consider tuning for better ranking")
        else:
            write("⚠ Retrieval system needs improvement")
            write("  - Review embedding model or retrieval strategy")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VTK RAG pipeline")
    parser.add_argument(
        '--test-set',
        type=Path,
        default=Path('data/processed/test_set.jsonl'),
        help='Test set file'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=10,
        help='Number of examples to evaluate'
    )
    parser.add_argument(
        '--mode',
        choices=['retrieval', 'end-to-end'],
        default='retrieval',
        help='Evaluation mode'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evaluation/eval_results.json'),
        help='Output report file'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable code validation (only for end-to-end mode)'
    )
    parser.add_argument(
        '--enable-visual-testing',
        action='store_true',
        help='Enable visual testing (executes code in Docker, requires RUN_VISUAL_TESTS=1)'
    )
    
    args = parser.parse_args()
    
    # VTK RAG Evaluation
    
    # Load test set
    if not args.test_set.exists():
        print(f"\n❌ Error: Test set not found: {args.test_set}")
        print("Run test_set_builder.py first to generate test set")
        return 1
    
    evaluator = RAGEvaluator(
        enable_llm=(args.mode == 'end-to-end'),
        enable_validation=not args.no_validation,
        enable_visual_testing=args.enable_visual_testing
    )
    test_set = evaluator.load_test_set(args.test_set)
    
    print(f"Loaded {len(test_set)} examples | Evaluating {args.num_examples} | Mode: {args.mode}")
    
    # Run evaluation
    test_subset = test_set[:args.num_examples]
    
    if args.mode == 'retrieval':
        results = evaluator.evaluate_retrieval_only(test_subset)
    else:
        results = evaluator.evaluate_end_to_end(test_subset)
    
    # Generate report
    evaluator.generate_report(results, args.output, mode=args.mode)
    
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
