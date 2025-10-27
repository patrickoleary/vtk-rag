#!/usr/bin/env python3
"""
End-to-End Metrics for VTK RAG

Evaluates complete pipeline output quality:
- Exactness: How closely generated code matches gold standard
- Completeness: Does response include all necessary components
- Attributable: Are claims backed by retrieved sources

Measures final answer quality after LLM generation and post-processing.
"""

import re
from typing import List, Dict, Set, Any
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class EndToEndMetrics:
    """Container for end-to-end metrics focused on code quality"""
    # Code Quality Metrics
    code_exactness: float          # Similarity to gold standard (0-1)
    code_correctness: float        # Has all necessary VTK components (0-1)
    code_syntax_valid: float       # % with valid Python syntax
    
    # Explanation Quality
    explanation_present: float     # % with any explanation
    
    # Validation/Error Metrics
    validation_attempted: float = 0.0  # % that went through validation
    validation_errors_found: float = 0.0  # % that had initial errors
    validation_needed_retry: float = 0.0  # % that needed fixes
    validation_success_rate: float = 0.0  # % that passed after fixes
    validation_avg_retries: float = 0.0  # Average fix attempts per code
    
    # Security Validation Metrics (NEW)
    security_check_performed: float = 0.0  # % that went through security check
    security_check_passed: float = 0.0  # % that passed security validation
    security_issues_found: float = 0.0  # % that had security issues detected
    
    # API Validation Metrics (NEW)
    api_validation_attempted: float = 0.0  # % that went through API validation
    api_errors_detected: float = 0.0  # % that had API errors
    api_errors_fixed: float = 0.0  # % of errors successfully fixed by LLM
    api_errors_unfixed: float = 0.0  # % of errors that remain unfixed
    api_avg_errors_per_query: float = 0.0  # Average API errors per query
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'code_exactness': self.code_exactness,
            'code_correctness': self.code_correctness,
            'code_syntax_valid': self.code_syntax_valid,
            'explanation_present': self.explanation_present,
            'validation_attempted': self.validation_attempted,
            'validation_errors_found': self.validation_errors_found,
            'validation_needed_retry': self.validation_needed_retry,
            'validation_success_rate': self.validation_success_rate,
            'validation_avg_retries': self.validation_avg_retries,
            'security_check_performed': self.security_check_performed,
            'security_check_passed': self.security_check_passed,
            'security_issues_found': self.security_issues_found,
            'api_validation_attempted': self.api_validation_attempted,
            'api_errors_detected': self.api_errors_detected,
            'api_errors_fixed': self.api_errors_fixed,
            'api_errors_unfixed': self.api_errors_unfixed,
            'api_avg_errors_per_query': self.api_avg_errors_per_query
        }


class EndToEndEvaluator:
    """
    Evaluate end-to-end RAG pipeline quality
    
    Measures final output after retrieval → grounding → generation → post-processing
    """
    
    def __init__(self):
        self.vtk_class_pattern = r'\b(vtk[A-Z]\w+)\b'
    
    def code_exactness(self, generated_code: str, gold_code: str) -> float:
        """
        Measure code exactness using normalized string similarity
        
        Uses SequenceMatcher for fuzzy matching since:
        - Variable names may differ
        - Comments may be added/modified
        - Formatting may vary
        
        Args:
            generated_code: Generated code from LLM
            gold_code: Gold standard code
        
        Returns:
            Exactness score [0.0, 1.0]
        """
        if not generated_code or not gold_code:
            return 0.0
        
        # Normalize: remove comments, extra whitespace
        gen_norm = self._normalize_code(generated_code)
        gold_norm = self._normalize_code(gold_code)
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, gen_norm, gold_norm).ratio()
        
        return similarity
    
    def code_completeness(self, generated_code: str, gold_code: str) -> float:
        """
        Measure code completeness - does it have necessary components?
        
        Checks for:
        - Required VTK classes (core functionality)
        - Imports (code structure)
        - Key method calls (with pythonic API support)
        
        Args:
            generated_code: Generated code from LLM
            gold_code: Gold standard code
        
        Returns:
            Completeness score [0.0, 1.0]
        """
        if not generated_code or not gold_code:
            return 0.0
        
        score = 0.0
        checks = 0
        
        # Check 1: Has required VTK classes
        gold_classes = set(re.findall(self.vtk_class_pattern, gold_code))
        gen_classes = set(re.findall(self.vtk_class_pattern, generated_code))
        
        if gold_classes:
            checks += 1
            overlap = len(gold_classes.intersection(gen_classes))
            score += overlap / len(gold_classes)
        
        # Check 2: Has imports
        has_gold_imports = 'import' in gold_code or 'from' in gold_code
        has_gen_imports = 'import' in generated_code or 'from' in generated_code
        
        if has_gold_imports:
            checks += 1
            score += 1.0 if has_gen_imports else 0.0
        
        # Check 3: Key method calls present (pythonic API aware)
        # Extract method calls like .SetRadius(), .Update()
        gold_methods = set(re.findall(r'\.([A-Z]\w+)\(', gold_code))
        gen_methods = set(re.findall(r'\.([A-Z]\w+)\(', generated_code))
        
        # Handle pythonic API: constructor params, attribute assignments, and >> operator
        # Pattern 1: vtkActor(mapper=x) replaces .SetMapper()
        # Pattern 2: obj.render_window = x replaces .SetRenderWindow()
        # Pattern 3: source >> mapper replaces .SetInputConnection() and .GetOutputPort()
        pythonic_constructor_patterns = {
            'SetMapper': r'vtkActor\s*\([^)]*mapper\s*=',
            'SetRenderWindow': r'vtkRenderWindowInteractor\s*\([^)]*render_window\s*=',
        }
        
        pythonic_attribute_patterns = {
            'SetRenderWindow': r'\.render_window\s*=',
            'SetMapper': r'\.mapper\s*=',
        }
        
        # Pipeline operator '>>' replaces SetInputConnection and GetOutputPort
        has_pipeline_operator = '>>' in generated_code
        pipeline_methods = {'SetInputConnection', 'GetOutputPort', 'SetInput', 'SetInputData'}
        
        # Give credit for pythonic API usage
        for method, pattern in pythonic_constructor_patterns.items():
            if method in gold_methods and method not in gen_methods:
                if re.search(pattern, generated_code):
                    gen_methods.add(method)
        
        for method, pattern in pythonic_attribute_patterns.items():
            if method in gold_methods and method not in gen_methods:
                if re.search(pattern, generated_code):
                    gen_methods.add(method)
        
        # Credit >> operator for pipeline methods
        if has_pipeline_operator:
            for method in pipeline_methods:
                if method in gold_methods and method not in gen_methods:
                    gen_methods.add(method)
        
        if gold_methods:
            checks += 1
            overlap = len(gold_methods.intersection(gen_methods))
            score += overlap / len(gold_methods)
        
        return score / checks if checks > 0 else 0.0
    
    def explanation_completeness(self, explanation: str, code: str) -> float:
        """
        Measure explanation completeness based on explanation-to-code ratio
        
        A good explanation should be longer than the code itself to provide
        meaningful context and understanding.
        
        Scoring:
        - <1x code length: poor (60%)
        - 1-2x: ok (70%)
        - 2-3x: good (80%)
        - 3-4x: great (90%)
        - 4-5x+: perfect (100%)
        
        Args:
            explanation: Generated explanation text
            code: Generated code to compare against
        
        Returns:
            Completeness score [0.0, 1.0]
        """
        if not explanation:
            return 0.0
        
        if not code:
            # If no code, just check if explanation exists
            return 1.0 if len(explanation) > 50 else 0.0
        
        # Calculate ratio of explanation length to code length
        explanation_len = len(explanation.strip())
        code_len = len(code.strip())
        
        if code_len == 0:
            return 1.0 if explanation_len > 0 else 0.0
        
        ratio = explanation_len / code_len
        
        # Score based on ratio
        if ratio < 1.0:
            return 0.60  # poor
        elif ratio < 2.0:
            return 0.70  # ok
        elif ratio < 3.0:
            return 0.80  # good
        elif ratio < 4.0:
            return 0.90  # great
        else:
            return 1.00  # perfect
    
    def attributable(
        self,
        parsed_response: Any,
        retrieved_chunk_ids: Set[str]
    ) -> float:
        """
        Measure attributability - are claims backed by sources?
        
        Checks:
        - Has citations
        - Citations reference retrieved chunks
        - Key statements are cited
        
        Args:
            parsed_response: ParsedResponse object with citations
            retrieved_chunk_ids: Set of chunk IDs that were retrieved
        
        Returns:
            Attributability score [0.0, 1.0]
        """
        score = 0.0
        checks = 0
        
        # Check 1: Has citations
        checks += 1
        has_citations = parsed_response.has_citations
        score += 1.0 if has_citations else 0.0
        
        # Check 2: Citations are valid (reference retrieved chunks)
        if has_citations:
            checks += 1
            valid_citations = sum(
                1 for cite in parsed_response.citations
                if cite.chunk_id in retrieved_chunk_ids
            )
            score += valid_citations / len(parsed_response.citations) if parsed_response.citations else 0.0
        
        # Check 3: VTK classes are cited
        vtk_classes = parsed_response.get_all_vtk_classes()
        if vtk_classes and has_citations:
            checks += 1
            # At least some VTK usage should be cited
            answer_text = parsed_response.raw_text
            
            # Count how many class mentions are near citations
            cited_classes = 0
            for cls in vtk_classes:
                # Check if class appears near a citation [N]
                pattern = rf'{cls}.*?\[\d+\]|\[\d+\].*?{cls}'
                if re.search(pattern, answer_text, re.DOTALL):
                    cited_classes += 1
            
            score += cited_classes / len(vtk_classes) if vtk_classes else 0.0
        
        return score / checks if checks > 0 else 0.0
    
    def citation_precision(
        self,
        parsed_response: Any,
        retrieved_chunk_ids: Set[str]
    ) -> float:
        """
        Citation precision - fraction of citations that are valid
        
        Args:
            parsed_response: ParsedResponse object
            retrieved_chunk_ids: Set of valid chunk IDs
        
        Returns:
            Precision [0.0, 1.0]
        """
        if not parsed_response.citations:
            return 0.0
        
        valid = sum(
            1 for cite in parsed_response.citations
            if cite.chunk_id in retrieved_chunk_ids
        )
        
        return valid / len(parsed_response.citations)
    
    def citation_recall(
        self,
        parsed_response: Any,
        relevant_chunk_ids: Set[str]
    ) -> float:
        """
        Citation recall - fraction of relevant sources cited
        
        Args:
            parsed_response: ParsedResponse object
            relevant_chunk_ids: Set of relevant chunk IDs (gold standard)
        
        Returns:
            Recall [0.0, 1.0]
        """
        if not relevant_chunk_ids:
            return 0.0
        
        cited_chunks = set(c.chunk_id for c in parsed_response.citations if c.chunk_id)
        cited_relevant = cited_chunks.intersection(relevant_chunk_ids)
        
        return len(cited_relevant) / len(relevant_chunk_ids)
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        # Remove empty lines
        code = code.strip()
        return code
    
    def data_file_extraction(
        self,
        parsed_response: Any,
        expected_data_files: List[str]
    ) -> float:
        """
        Check if expected data files were extracted from metadata
        
        Args:
            parsed_response: ParsedResponse object
            expected_data_files: Expected data files from test example
        
        Returns:
            1.0 if all expected files found, 0.0 otherwise, -1.0 if N/A
        """
        if not expected_data_files:
            return -1.0  # N/A - no files expected
        
        extracted_files = [df.filename for df in parsed_response.data_files]
        
        # Check if all expected files are in extracted files
        all_found = all(expected in extracted_files for expected in expected_data_files)
        
        return 1.0 if all_found else 0.0
    
    def download_url_formatting(
        self,
        parsed_response: Any,
        expected_data_files: List[str]
    ) -> float:
        """
        Check if download URLs are properly formatted for data files
        
        Args:
            parsed_response: ParsedResponse object
            expected_data_files: Expected data files from test example
        
        Returns:
            1.0 if URLs properly formatted, 0.0 otherwise, -1.0 if N/A
        """
        if not expected_data_files:
            return -1.0  # N/A
        
        # Check if data section can be formatted
        data_section = parsed_response.format_data_section()
        
        if not data_section:
            return 0.0
        
        # Check for key download components in formatted section
        has_curl = 'curl -o' in data_section
        has_wget = 'wget -O' in data_section
        has_url = 'http' in data_section
        
        return 1.0 if (has_curl and has_wget and has_url) else 0.0
    
    def baseline_image_extraction(
        self,
        parsed_response: Any,
        expected_has_image: bool
    ) -> float:
        """
        Check if result images were extracted when available
        
        This checks if visualization result images (from either examples or tests)
        were properly extracted and included in the response.
        
        Args:
            parsed_response: ParsedResponse object
            expected_has_image: Whether any result image is expected (image_url OR baseline_image)
        
        Returns:
            1.0 if correct, 0.0 if incorrect, -1.0 if N/A
        """
        if not expected_has_image:
            return -1.0  # N/A - no image expected
        
        has_baseline = parsed_response.has_baseline_images()
        
        return 1.0 if has_baseline else 0.0
    
    def evaluate_response(
        self,
        generated_code: str,
        gold_code: str,
        generated_explanation: str,
        parsed_response: Any,
        retrieved_chunk_ids: Set[str],
        relevant_chunk_ids: Set[str],
        expected_data_files: List[str] = None,
        expected_has_image: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single response
        
        Args:
            generated_code: Code from LLM
            gold_code: Gold standard code
            generated_explanation: Explanation from LLM
            parsed_response: ParsedResponse object
            retrieved_chunk_ids: Chunks retrieved by pipeline
            relevant_chunk_ids: Gold standard relevant chunks
            expected_data_files: Expected data files from test example
            expected_has_image: Whether result image expected (image_url OR baseline_image)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'code_exactness': self.code_exactness(generated_code, gold_code),
            'code_completeness': self.code_completeness(generated_code, gold_code),
            'explanation_completeness': self.explanation_completeness(generated_explanation, generated_code),
            'attributable': self.attributable(parsed_response, retrieved_chunk_ids),
            'citation_precision': self.citation_precision(parsed_response, retrieved_chunk_ids),
            'citation_recall': self.citation_recall(parsed_response, relevant_chunk_ids)
        }
        
        # Add augmented metadata extraction metrics
        if expected_data_files is not None:
            metrics['data_file_extraction'] = self.data_file_extraction(parsed_response, expected_data_files)
            metrics['download_url_formatting'] = self.download_url_formatting(parsed_response, expected_data_files)
        
        metrics['baseline_image_extraction'] = self.baseline_image_extraction(parsed_response, expected_has_image)
        
        return metrics
    
    def evaluate_test_set(
        self,
        results: List[Dict[str, Any]]
    ) -> EndToEndMetrics:
        """
        Evaluate over entire test set
        
        Args:
            results: List of evaluation results
        
        Returns:
            Averaged metrics
        """
        all_metrics = []
        
        for result in results:
            metrics = self.evaluate_response(
                result['generated_code'],
                result['gold_code'],
                result['generated_explanation'],
                result['parsed_response'],
                set(result['retrieved_chunk_ids']),
                set(result['relevant_chunk_ids']),
                expected_data_files=result.get('expected_data_files', []),
                expected_has_image=result.get('expected_has_image', False)
            )
            all_metrics.append(metrics)
        
        # Average code/citation metrics
        n = len(all_metrics)
        if n == 0:
            return EndToEndMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        avg = {
            key: sum(m[key] for m in all_metrics) / n
            for key in all_metrics[0].keys()
        }
        
        # Calculate validation metrics
        total_examples = len(results)
        attempted = sum(1 for r in results if r.get('validation_attempted', False))
        errors_found = sum(1 for r in results if r.get('validation_errors_found', 0) > 0)
        needed_retry = sum(1 for r in results if r.get('validation_retries', 0) > 0)
        passed_final = sum(1 for r in results if r.get('validation_final_status') == 'passed')
        total_retries = sum(r.get('validation_retries', 0) for r in results)
        
        validation_attempted = attempted / total_examples if total_examples > 0 else 0.0
        validation_errors_found = errors_found / attempted if attempted > 0 else 0.0
        validation_needed_retry = needed_retry / attempted if attempted > 0 else 0.0
        validation_success_rate = passed_final / errors_found if errors_found > 0 else 0.0
        validation_avg_retries = total_retries / total_examples if total_examples > 0 else 0.0
        
        # Check syntax validity
        syntax_valid = sum(1 for r in results if r.get('validation_final_status') in ['passed', 'not_run'])
        code_syntax_valid = syntax_valid / total_examples if total_examples > 0 else 0.0
        
        # Calculate security validation metrics (NEW)
        security_checked = sum(1 for r in results if 'security_check_passed' in r)
        security_passed = sum(1 for r in results if r.get('security_check_passed', True))
        security_failed = sum(1 for r in results if r.get('security_check_passed') == False)
        
        security_check_performed = security_checked / total_examples if total_examples > 0 else 0.0
        security_check_passed = security_passed / security_checked if security_checked > 0 else 1.0
        security_issues_found = security_failed / security_checked if security_checked > 0 else 0.0
        
        # Calculate API validation metrics (NEW)
        api_attempted = sum(1 for r in results if r.get('api_validation_attempted', False))
        api_errors_detected_count = sum(1 for r in results if r.get('api_validation_errors_count', 0) > 0)
        api_errors_fixed_count = sum(1 for r in results if r.get('api_validation_fixed', False))
        api_errors_unfixed_count = sum(1 for r in results if r.get('api_validation_errors_count', 0) > 0 and not r.get('api_validation_passed', True))
        total_api_errors = sum(r.get('api_validation_errors_count', 0) for r in results)
        
        api_validation_attempted = api_attempted / total_examples if total_examples > 0 else 0.0
        api_errors_detected = api_errors_detected_count / total_examples if total_examples > 0 else 0.0
        api_errors_fixed = api_errors_fixed_count / api_errors_detected_count if api_errors_detected_count > 0 else 0.0
        api_errors_unfixed = api_errors_unfixed_count / api_errors_detected_count if api_errors_detected_count > 0 else 0.0
        api_avg_errors_per_query = total_api_errors / total_examples if total_examples > 0 else 0.0
        
        return EndToEndMetrics(
            code_exactness=avg['code_exactness'],
            code_correctness=avg['code_completeness'],  # Rename to correctness
            code_syntax_valid=code_syntax_valid,
            explanation_present=avg['explanation_completeness'],
            validation_attempted=validation_attempted,
            validation_errors_found=validation_errors_found,
            validation_needed_retry=validation_needed_retry,
            validation_success_rate=validation_success_rate,
            validation_avg_retries=validation_avg_retries,
            security_check_performed=security_check_performed,
            security_check_passed=security_check_passed,
            security_issues_found=security_issues_found,
            api_validation_attempted=api_validation_attempted,
            api_errors_detected=api_errors_detected,
            api_errors_fixed=api_errors_fixed,
            api_errors_unfixed=api_errors_unfixed,
            api_avg_errors_per_query=api_avg_errors_per_query
        )
    
    def print_metrics(self, metrics: EndToEndMetrics):
        """Print metrics in readable format"""
        print("\nEnd-to-End Metrics:")
        print("-" * 40)
        print("Code Quality:")
        print(f"  Exactness:     {metrics.code_exactness:.1%} (similarity to gold)")
        print(f"  Correctness:   {metrics.code_correctness:.1%} (has all components)")
        print(f"  Syntax Valid:  {metrics.code_syntax_valid:.1%} (no syntax errors)")
        print()
        print("Explanation:")
        print(f"  Quality:       {metrics.explanation_present:.1%}")
        print()
        print("Syntax Validation Metrics:")
        print(f"  Attempted:     {metrics.validation_attempted:.1%}")
        print(f"  Errors Found:  {metrics.validation_errors_found:.1%}")
        print(f"  Needed Retry:  {metrics.validation_needed_retry:.1%}")
        print(f"  Success Rate:  {metrics.validation_success_rate:.1%}")
        print(f"  Avg Retries:   {metrics.validation_avg_retries:.2f}")
        print()
        print("Security Validation Metrics:")
        print(f"  Checked:       {metrics.security_check_performed:.1%} (% with security check)")
        print(f"  Passed:        {metrics.security_check_passed:.1%} (% that passed)")
        print(f"  Issues Found:  {metrics.security_issues_found:.1%} (% with security issues)")
        print()
        print("API Validation Metrics (VTK API Hallucination Detection):")
        print(f"  Attempted:     {metrics.api_validation_attempted:.1%} (% validated)")
        print(f"  Errors Found:  {metrics.api_errors_detected:.1%} (% with API errors)")
        print(f"  Errors Fixed:  {metrics.api_errors_fixed:.1%} (% of errors fixed by LLM)")
        print(f"  Errors Unfixed:{metrics.api_errors_unfixed:.1%} (% of errors remaining)")
        print(f"  Avg Errors:    {metrics.api_avg_errors_per_query:.2f} (per query)")


def main():
    """Test end-to-end metrics"""
    print("=" * 80)
    print("End-to-End Metrics - Test")
    print("=" * 80)
    
    evaluator = EndToEndEvaluator()
    
    # Test code exactness
    print("\nTest 1: Code Exactness")
    gold = "cylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)\ncylinder.Update()"
    generated = "cylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)\ncylinder.Update()"
    
    exactness = evaluator.code_exactness(generated, gold)
    print(f"  Identical code: {exactness:.3f} (expected: 1.0)")
    
    # Test code completeness
    print("\nTest 2: Code Completeness")
    gold = "from vtkmodules.vtkFiltersSources import vtkCylinderSource\ncylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)"
    generated = "cylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)"  # Missing import
    
    completeness = evaluator.code_completeness(generated, gold)
    print(f"  Missing imports: {completeness:.3f} (expected: < 1.0)")
    
    print("\n" + "=" * 80)
    print("✓ Tests complete")


if __name__ == '__main__':
    main()
