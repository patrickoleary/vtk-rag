#!/usr/bin/env python3
"""
Query Generator for VTK RAG System

Auto-generates explanation_queries from code_queries using
sophisticated transformation patterns.

Part of redesigned chunking strategy - creates parallel query sets.
"""

import re
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    Generate explanation queries from code queries
    
    Features:
    - Pattern-based transformations
    - VTK class-specific queries
    - Concept-based queries
    - Deduplication
    """
    
    # Transformation patterns (order matters - more specific first)
    TRANSFORMATIONS = [
        # Question transformations
        ("How do I create", "What is"),
        ("How do I", "What is"),
        ("How to create", "Understanding"),
        ("How to", "How does"),
        ("How can I", "What is"),
        
        # Action to concept
        ("Show me code to", "Explain how to"),
        ("Show me code for", "Explain the concept of"),
        ("Show me", "Explain"),
        ("Write code to", "Learn about"),
        ("Write code for", "Understanding"),
        
        # Specific to general
        ("code for", "concept of"),
        ("code to", "approach to"),
        ("example", "explanation"),
        ("example of", "explanation of"),
        ("sample", "overview"),
        
        # Implementation to understanding
        ("Create", "Understanding"),
        ("Generate", "Learn about"),
        ("Build", "Concept of"),
        ("Make", "Understanding"),
        ("Implement", "How does"),
        
        # VTK-specific
        ("render", "rendering concept"),
        ("visualize", "visualization concept"),
        ("filter", "filtering concept"),
    ]
    
    # Question templates for VTK classes
    CLASS_QUERY_TEMPLATES = [
        "What is {class_name}?",
        "Explain {class_name}",
        "When to use {class_name}?",
        "What does {class_name} do?",
        "{class_name} overview",
        "Understanding {class_name}",
    ]
    
    # Question templates for concepts
    CONCEPT_QUERY_TEMPLATES = [
        "Explain {concept} in VTK",
        "What is {concept}?",
        "How does {concept} work in VTK?",
        "VTK {concept} tutorial",
        "Understanding {concept} in VTK",
        "{concept} concept",
    ]
    
    # Method-specific templates
    METHOD_QUERY_TEMPLATES = [
        "What does {method} do?",
        "Explain {method} method",
        "How to use {method}?",
        "When to use {method}?",
    ]
    
    def generate_explanation_queries(
        self,
        code_queries: List[str],
        vtk_classes: List[str] = None,
        concepts: List[str] = None,
        methods: List[str] = None,
        max_queries: int = 20
    ) -> List[str]:
        """
        Generate explanation queries from code queries
        
        Args:
            code_queries: Original code-focused queries
            vtk_classes: VTK classes mentioned (e.g., vtkCylinderSource)
            concepts: VTK concepts (e.g., rendering, filtering)
            methods: Method names (e.g., SetRadius, Update)
            max_queries: Maximum number of queries to generate
        
        Returns:
            List of explanation-focused queries (deduplicated)
        """
        explanation_queries = set()
        
        # Transform code queries
        for query in code_queries:
            transformed = self._transform_query(query)
            if transformed and transformed != query:
                explanation_queries.add(transformed)
        
        # Add class-based queries
        if vtk_classes:
            for cls in vtk_classes[:3]:  # Limit to top 3
                for template in self.CLASS_QUERY_TEMPLATES[:4]:  # Use 4 templates
                    explanation_queries.add(template.format(class_name=cls))
        
        # Add concept-based queries
        if concepts:
            for concept in concepts[:3]:  # Limit to top 3
                for template in self.CONCEPT_QUERY_TEMPLATES[:3]:  # Use 3 templates
                    explanation_queries.add(template.format(concept=concept))
        
        # Add method-based queries
        if methods:
            for method in methods[:2]:  # Limit to top 2
                for template in self.METHOD_QUERY_TEMPLATES[:2]:  # Use 2 templates
                    explanation_queries.add(template.format(method=method))
        
        # Convert to list and limit
        result = sorted(list(explanation_queries))[:max_queries]
        
        return result
    
    def _transform_query(self, query: str) -> str:
        """Apply transformation patterns to a query"""
        transformed = query
        
        # Apply each transformation pattern
        for old_pattern, new_pattern in self.TRANSFORMATIONS:
            if old_pattern in transformed:
                transformed = transformed.replace(old_pattern, new_pattern)
                break  # Only apply first matching pattern
        
        return transformed
    
    def extract_vtk_classes_from_query(self, query: str) -> List[str]:
        """Extract VTK class names mentioned in query"""
        # Pattern: vtk followed by uppercase letter and word chars
        pattern = r'\b(vtk[A-Z]\w+)\b'
        matches = re.findall(pattern, query)
        return list(set(matches))
    
    def extract_methods_from_query(self, query: str) -> List[str]:
        """Extract method names from query"""
        # Common VTK method patterns
        patterns = [
            r'\b(Set\w+)\b',
            r'\b(Get\w+)\b',
            r'\b(Update)\b',
            r'\b(Render)\b',
            r'\b(Add\w+)\b',
            r'\b(Remove\w+)\b',
        ]
        
        methods = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            methods.extend(matches)
        
        return list(set(methods))


class QueryEnhancer:
    """Enhance and expand query sets"""
    
    @staticmethod
    def add_variations(queries: List[str]) -> List[str]:
        """Add natural variations to queries"""
        variations = set(queries)
        
        for query in queries:
            query_lower = query.lower()
            
            # Add question mark if missing
            if not query.endswith('?'):
                variations.add(query + '?')
            
            # Add VTK prefix if discussing VTK concepts
            if 'vtk' not in query_lower and any(word in query_lower for word in ['render', 'visual', 'filter', 'map']):
                variations.add(f"VTK {query}")
        
        return sorted(list(variations))
    
    @staticmethod
    def prioritize_queries(queries: List[str]) -> List[str]:
        """Sort queries by likely importance"""
        def query_priority(query: str) -> int:
            """Higher number = higher priority"""
            priority = 0
            
            # Prioritize "What is" questions (fundamental)
            if query.startswith("What is"):
                priority += 10
            
            # Prioritize "Explain" questions
            if query.startswith("Explain"):
                priority += 8
            
            # Prioritize questions with VTK classes
            if re.search(r'\bvtk[A-Z]\w+', query):
                priority += 5
            
            # Deprioritize very long queries
            if len(query) > 100:
                priority -= 3
            
            return priority
        
        return sorted(queries, key=query_priority, reverse=True)


def main():
    """Test the query generator"""
    print("=" * 80)
    print("QueryGenerator - Test")
    print("=" * 80)
    
    # Test data
    test_cases = [
        {
            'name': 'Simple geometry example',
            'code_queries': [
                'How do I create a cylinder in VTK?',
                'Show me code to render a 3D cylinder',
                'VTK cylinder example'
            ],
            'vtk_classes': ['vtkCylinderSource', 'vtkPolyDataMapper', 'vtkActor'],
            'concepts': ['rendering', 'geometry', 'sources'],
            'methods': ['SetRadius', 'SetHeight', 'Update']
        },
        {
            'name': 'Data I/O example',
            'code_queries': [
                'How to read a CSV file in VTK?',
                'Show me code for reading delimited text'
            ],
            'vtk_classes': ['vtkDelimitedTextReader'],
            'concepts': ['io', 'data'],
            'methods': ['SetFileName', 'Update']
        }
    ]
    
    generator = QueryGenerator()
    enhancer = QueryEnhancer()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        
        print(f"\nOriginal code queries:")
        for q in test_case['code_queries']:
            print(f"  • {q}")
        
        # Generate explanation queries
        explanation_queries = generator.generate_explanation_queries(
            code_queries=test_case['code_queries'],
            vtk_classes=test_case.get('vtk_classes'),
            concepts=test_case.get('concepts'),
            methods=test_case.get('methods'),
            max_queries=15
        )
        
        print(f"\nGenerated explanation queries ({len(explanation_queries)}):")
        for q in explanation_queries[:10]:  # Show first 10
            print(f"  • {q}")
        
        if len(explanation_queries) > 10:
            print(f"  ... and {len(explanation_queries) - 10} more")
        
        # Test enhancements
        enhanced = enhancer.add_variations(explanation_queries[:5])
        prioritized = enhancer.prioritize_queries(enhanced)
        
        print(f"\nTop 5 prioritized queries:")
        for q in prioritized[:5]:
            print(f"  • {q}")
    
    print("\n" + "=" * 80)
    print("✓ QueryGenerator test complete")
    print("\nKey features demonstrated:")
    print("  • Pattern-based query transformation")
    print("  • VTK class-specific questions")
    print("  • Concept-based questions")
    print("  • Method-based questions")
    print("  • Query variations and prioritization")


if __name__ == '__main__':
    main()
