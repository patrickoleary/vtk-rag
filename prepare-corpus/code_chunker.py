#!/usr/bin/env python3
"""
Code-Only Chunker for VTK RAG System

Extracts pure code from examples/tests, strips explanatory text,
generates small focused chunks (200-600 tokens) with rich metadata.

Part of redesigned chunking strategy - separates CODE from EXPLANATION.
"""

import re
import ast
import warnings
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for code chunking"""
    target_tokens: int = 400      # Sweet spot for sequential thinking
    min_tokens: int = 200          # Don't fragment too small
    max_tokens: int = 600          # Hard limit
    chars_per_token: float = 4.0   # Approximation


@dataclass
class CodeChunk:
    """Represents a code-only chunk"""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content: str  # Pure code only
    content_type: str = "code"
    metadata: Dict[str, Any] = None
    source_type: str = "example"
    original_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeExtractor:
    """Extract and clean code from examples"""
    
    @staticmethod
    def extract_imports(code: str) -> str:
        """Extract all import statements"""
        lines = code.split('\n')
        imports = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(line)
        
        return '\n'.join(imports)
    
    @staticmethod
    def extract_functions(code: str) -> List[Tuple[str, int, int]]:
        """
        Extract function definitions with their line positions
        Returns: List of (function_name, start_line, end_line)
        """
        try:
            # Suppress SyntaxWarning from raw VTK test data
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                tree = ast.parse(code)
            
            functions = []
            lines = code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1
                    
                    # Find end line (next function/class or EOF)
                    end_line = len(lines)
                    for other_node in ast.walk(tree):
                        if (isinstance(other_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and other_node != node
                            and other_node.lineno > node.lineno):
                            end_line = min(end_line, other_node.lineno - 1)
                    
                    functions.append((node.name, start_line, end_line))
            
            return functions
        except SyntaxError:
            logger.warning("Could not parse code for functions")
            return []
    
    @staticmethod
    def strip_comments_and_docstrings(code: str) -> str:
        """Remove comments and docstrings, keeping essential code"""
        try:
            # Suppress SyntaxWarning from raw VTK test data
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                tree = ast.parse(code)
            
            # Remove docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (ast.get_docstring(node)):
                        # Get first statement (docstring)
                        if node.body and isinstance(node.body[0], ast.Expr):
                            if isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                                node.body = node.body[1:]  # Remove docstring
            
            # Convert back to code
            code = ast.unparse(tree)
        except:
            pass  # Keep original if unparsing fails
        
        # Remove inline comments
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            # Keep line up to # (but not in strings)
            if '#' in line:
                # Simple heuristic: keep if # is in quotes
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        line = line[:i].rstrip()
                        break
            
            if line.strip():  # Keep non-empty lines
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    @staticmethod
    def remove_print_statements(code: str) -> str:
        """Remove print statements (noise for code examples)"""
        lines = code.split('\n')
        cleaned = []
        
        for line in lines:
            if 'print(' not in line:
                cleaned.append(line)
        
        return '\n'.join(cleaned)


class ImportStyleDetector:
    """Detect VTK import style"""
    
    @staticmethod
    def detect_import_style(code: str) -> str:
        """
        Detect import pattern (NOT API syntax style)
        
        Returns:
            'modular': Selective module imports:
                       - from vtkmodules.vtkXxx import Y
                       - import vtkmodules.vtkXxx
            'monolithic': import vtk OR from vtkmodules.all (loads everything)
            'mixed': Uses both modular and monolithic (bad practice)
            'none': No VTK imports (helper functions, snippets)
        
        Note: vtkmodules.all is monolithic because it loads everything,
              just with different syntax than 'import vtk'
        """
        # Detect modular patterns:
        # 1. from vtkmodules.vtkXxx import ... (but NOT vtkmodules.all)
        has_modular_from = bool(re.search(r'from\s+vtkmodules\.vtk\w+\s+import', code))
        # 2. import vtkmodules.vtkXxx (backend enabling and module imports)
        has_modular_import = bool(re.search(r'import\s+vtkmodules\.vtk\w+', code))
        has_modular = has_modular_from or has_modular_import
        
        # Detect monolithic patterns
        has_import_vtk = bool(re.search(r'import\s+vtk\b', code))
        has_vtkmodules_all = bool(re.search(r'from\s+vtkmodules\.all\s+import', code))
        has_monolithic = has_import_vtk or has_vtkmodules_all
        
        # Check for mixing (bad practice)
        if has_modular and has_monolithic:
            return 'mixed'
        elif has_modular:
            return 'modular'
        elif has_monolithic:
            return 'monolithic'
        else:
            return 'none'
    
    @staticmethod
    def detect_source_style(example: Dict[str, Any], cleaned_code: str) -> str:
        """
        Detect API syntax style (NOT import pattern)
        
        Args:
            example: Original example dict with metadata
            cleaned_code: Cleaned code after comment removal
        
        Returns:
            'pythonic': Uses pythonic API (properties, >>, keyword args)
            'basic': Uses traditional API (Set/Get methods)
            'snippet': Helper function/utility without complete example context
        
        Detection:
            - "Pythonic" in title → pythonic
            - Starts with "def" and no VTK imports → snippet  
            - Otherwise → basic
        """
        example_id = example.get('id', '')
        title = example.get('title', '')
        
        # Check for Pythonic in title/id - this indicates API style
        if 'Pythonic' in example_id or 'Pythonic' in title:
            return 'pythonic'
        
        # Check if this is a code snippet (helper function without VTK imports)
        # These are utility functions meant to be used in larger examples
        # Look at cleaned code (not original) to ignore commented-out imports
        code_lines = [l.strip() for l in cleaned_code.split('\n') if l.strip()]
        starts_with_def = code_lines and code_lines[0].startswith('def ')
        
        # Check for VTK imports (not just any imports)
        has_vtk_import = bool(re.search(r'^(import\s+(vtk|vtkmodules)|from\s+(vtk|vtkmodules))', cleaned_code, re.MULTILINE))
        
        if starts_with_def and not has_vtk_import:
            return 'snippet'
        
        return 'basic'


class VTKAnalyzer:
    """Analyze VTK-specific aspects of code"""
    
    @staticmethod
    def extract_vtk_classes(code: str) -> List[str]:
        """Extract all VTK class names used in code"""
        # Pattern: vtk followed by uppercase letter and word chars
        pattern = r'\b(vtk[A-Z]\w+)\b'
        matches = re.findall(pattern, code)
        return sorted(list(set(matches)))
    
    @staticmethod
    def detect_features(code: str, vtk_classes: List[str]) -> Dict[str, bool]:
        """Detect what VTK features the code uses"""
        code_lower = code.lower()
        
        # Visualization keywords
        viz_classes = ['vtkRenderer', 'vtkRenderWindow', 'vtkActor', 
                       'vtkRenderWindowInteractor', 'vtkCamera']
        has_visualization = any(cls in code for cls in viz_classes)
        
        # Data I/O keywords
        io_patterns = ['Reader', 'Writer', 'Importer', 'Exporter']
        has_data_io = any(pattern in cls for cls in vtk_classes for pattern in io_patterns)
        
        # Filters
        has_filters = any('Filter' in cls for cls in vtk_classes)
        
        # Sources
        has_sources = any('Source' in cls for cls in vtk_classes)
        
        # Mappers
        has_mappers = any('Mapper' in cls for cls in vtk_classes)
        
        return {
            'has_visualization': has_visualization,
            'has_data_io': has_data_io,
            'has_filters': has_filters,
            'has_sources': has_sources,
            'has_mappers': has_mappers
        }
    
    @staticmethod
    def calculate_complexity(code: str, vtk_classes: List[str]) -> str:
        """
        Calculate code complexity
        
        Criteria (aggressive - prefer sequential decomposition):
        - >7 VTK classes = complex
        - Has I/O (Reader/Writer) = complex
        - >1 filter = complex
        - Otherwise: simple or moderate
        
        Returns: 'simple', 'moderate', or 'complex'
        """
        class_count = len(vtk_classes)
        
        # Check for I/O classes
        io_patterns = ['Reader', 'Writer', 'Importer', 'Exporter']
        has_io = any(pattern in cls for cls in vtk_classes for pattern in io_patterns)
        
        # Count filters
        filters = [cls for cls in vtk_classes if 'Filter' in cls]
        filter_count = len(filters)
        
        # Complex if:
        # 1. More than 7 VTK classes
        if class_count > 7:
            return 'complex'
        
        # 2. Has I/O operations
        if has_io:
            return 'complex'
        
        # 3. More than 1 filter
        if filter_count > 1:
            return 'complex'
        
        # Moderate: 4-7 classes OR 1 filter
        if class_count >= 4 or filter_count == 1:
            return 'moderate'
        
        # Simple: <=3 classes, no I/O, no filters
        return 'simple'
    
    @staticmethod
    def detect_requires_data_files(code: str, data_files: List[str]) -> bool:
        """
        Detect if code requires external data files
        """
        # If metadata has data_files, it requires them
        if data_files:
            return True
        
        # Look for file I/O patterns
        file_patterns = [
            r'SetFileName\(',
            r'\.read\(',
            r'\.open\(',
            r'with open\(',
        ]
        
        for pattern in file_patterns:
            if re.search(pattern, code):
                return True
        
        return False


class DataFileHandler:
    """Handle data file metadata"""
    
    @staticmethod
    def preserve_data_files(example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and preserve data file metadata"""
        return {
            'data_files': example.get('data_files', []),
            'data_download_info': example.get('data_download_info', [])
        }


class CodeOnlyChunker:
    """
    Main chunker for creating code-only chunks
    
    Features:
    - Extracts pure code (strips text/comments/docstrings)
    - Detects import style (pythonic/monolithic/mixed)
    - Detects source style (prefers Pythonic examples)
    - Extracts VTK classes used
    - Detects features (visualization, I/O, filters, etc.)
    - Calculates complexity
    - Preserves data file metadata
    - Target: 200-600 tokens per chunk
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.extractor = CodeExtractor()
        self.import_detector = ImportStyleDetector()
        self.vtk_analyzer = VTKAnalyzer()
        self.data_handler = DataFileHandler()
    
    def chunk_example(self, example: Dict[str, Any]) -> List[CodeChunk]:
        """
        Chunk an example into code-only chunks
        
        Args:
            example: Dict with 'code', 'id', 'title', 'code_queries', etc.
        
        Returns:
            List of CodeChunk objects
        """
        code = example.get('code', '')
        example_id = example.get('id', 'unknown')
        title = example.get('title', 'Untitled')
        code_queries = example.get('code_queries', [])
        
        if not code:
            logger.warning(f"No code in example {example_id}")
            return []
        
        # Step 1: Extract and clean code
        cleaned_code = self._clean_code(code)
        
        # Step 2: Detect metadata
        source_style = self.import_detector.detect_source_style(example, cleaned_code)
        import_style = self.import_detector.detect_import_style(cleaned_code)
        vtk_classes = self.vtk_analyzer.extract_vtk_classes(cleaned_code)
        features = self.vtk_analyzer.detect_features(cleaned_code, vtk_classes)
        complexity = self.vtk_analyzer.calculate_complexity(cleaned_code, vtk_classes)
        data_info = self.data_handler.preserve_data_files(example)
        requires_data_files = self.vtk_analyzer.detect_requires_data_files(
            cleaned_code, 
            data_info['data_files']
        )
        
        # Step 3: Chunk the code
        chunks = self._chunk_code(
            cleaned_code,
            example_id,
            title,
            source_style,
            code_queries,
            import_style,
            vtk_classes,
            features,
            complexity,
            data_info,
            requires_data_files
        )
        
        return chunks
    
    def _clean_code(self, code: str) -> str:
        """Clean code: remove comments, docstrings, print statements"""
        cleaned = code
        
        # Remove comments and docstrings
        cleaned = self.extractor.strip_comments_and_docstrings(cleaned)
        
        # Remove print statements (noise)
        cleaned = self.extractor.remove_print_statements(cleaned)
        
        # Remove extra blank lines
        lines = cleaned.split('\n')
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip consecutive blanks
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return '\n'.join(cleaned_lines)
    
    def _chunk_code(
        self,
        code: str,
        example_id: str,
        title: str,
        source_style: str,
        code_queries: List[str],
        import_style: str,
        vtk_classes: List[str],
        features: Dict[str, bool],
        complexity: str,
        data_info: Dict[str, Any],
        requires_data_files: bool
    ) -> List[CodeChunk]:
        """
        Split code into chunks of 200-600 tokens
        
        Strategy:
        - If code fits in one chunk: single chunk
        - Otherwise: split by functions, group small ones
        """
        code_lines = code.split('\n')
        total_chars = len(code)
        estimated_tokens = int(total_chars / self.config.chars_per_token)
        
        # Single chunk if small enough
        if estimated_tokens <= self.config.max_tokens:
            return [self._create_chunk(
                code,
                example_id,
                title,
                source_style,
                0,
                1,
                code_queries,
                import_style,
                vtk_classes,
                features,
                complexity,
                data_info,
                requires_data_files
            )]
        
        # Multiple chunks: split by functions
        imports = self.extractor.extract_imports(code)
        functions = self.extractor.extract_functions(code)
        
        if not functions:
            # No functions, split by line count
            return self._split_by_lines(
                code,
                example_id,
                title,
                source_style,
                code_queries,
                import_style,
                vtk_classes,
                features,
                complexity,
                data_info,
                requires_data_files
            )
        
        # Split by functions
        chunks = []
        current_func_group = []
        current_size = len(imports)
        
        for func_name, start, end in functions:
            func_code = '\n'.join(code_lines[start:end])
            func_size = len(func_code)
            
            if current_size + func_size > self.config.max_tokens * self.config.chars_per_token:
                # Create chunk from current group
                if current_func_group:
                    chunk_code = self._build_chunk_code(imports, current_func_group, code_lines)
                    chunks.append(self._create_chunk(
                        chunk_code,
                        example_id,
                        title,
                        source_style,
                        len(chunks),
                        0,  # Will update total later
                        code_queries,
                        import_style,
                        vtk_classes,
                        features,
                        complexity,
                        data_info,
                        requires_data_files
                    ))
                
                # Start new group
                current_func_group = [(func_name, start, end)]
                current_size = len(imports) + func_size
            else:
                current_func_group.append((func_name, start, end))
                current_size += func_size
        
        # Final chunk
        if current_func_group:
            chunk_code = self._build_chunk_code(imports, current_func_group, code_lines)
            chunks.append(self._create_chunk(
                chunk_code,
                example_id,
                title,
                source_style,
                len(chunks),
                0,
                code_queries,
                import_style,
                vtk_classes,
                features,
                complexity,
                data_info,
                requires_data_files
            ))
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _split_by_lines(
        self,
        code: str,
        example_id: str,
        title: str,
        source_style: str,
        code_queries: List[str],
        import_style: str,
        vtk_classes: List[str],
        features: Dict[str, bool],
        complexity: str,
        data_info: Dict[str, Any],
        requires_data_files: bool
    ) -> List[CodeChunk]:
        """Split code by line count when no functions detected"""
        lines = code.split('\n')
        max_lines = int(self.config.max_tokens * self.config.chars_per_token / 50)  # ~50 chars/line
        
        chunks = []
        start = 0
        while start < len(lines):
            end = min(start + max_lines, len(lines))
            chunk_code = '\n'.join(lines[start:end])
            
            chunks.append(self._create_chunk(
                chunk_code,
                example_id,
                title,
                source_style,
                len(chunks),
                0,
                code_queries,
                import_style,
                vtk_classes,
                features,
                complexity,
                data_info,
                requires_data_files
            ))
            
            start = end
        
        # Update total
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _build_chunk_code(
        self,
        imports: str,
        functions: List[Tuple[str, int, int]],
        code_lines: List[str]
    ) -> str:
        """Build chunk code from imports + functions"""
        parts = []
        
        if imports:
            parts.append(imports)
            parts.append('')  # Blank line
        
        for func_name, start, end in functions:
            func_code = '\n'.join(code_lines[start:end])
            parts.append(func_code)
            parts.append('')  # Blank line between functions
        
        return '\n'.join(parts).strip()
    
    def _create_chunk(
        self,
        code: str,
        example_id: str,
        title: str,
        source_style: str,
        chunk_index: int,
        total_chunks: int,
        code_queries: List[str],
        import_style: str,
        vtk_classes: List[str],
        features: Dict[str, bool],
        complexity: str,
        data_info: Dict[str, Any],
        requires_data_files: bool
    ) -> CodeChunk:
        """Create a CodeChunk object"""
        # Build chunk_id
        style_suffix = "Pythonic" if source_style == "pythonic" else ""
        chunk_id = f"{title}{style_suffix}_code_{chunk_index}"
        
        # Check if has imports
        has_imports = bool(self.extractor.extract_imports(code))
        
        # Count lines and functions
        line_count = len([l for l in code.split('\n') if l.strip()])
        function_count = len(self.extractor.extract_functions(code))
        
        # Build metadata
        metadata = {
            'title': title,
            'code_queries': code_queries,
            'source_style': source_style,
            'import_style': import_style,
            'vtk_classes': vtk_classes,
            'has_imports': has_imports,
            **features,  # has_visualization, has_data_io, etc.
            'complexity': complexity,
            'line_count': line_count,
            'function_count': function_count,
            'data_files': data_info['data_files'],
            'data_download_info': data_info['data_download_info'],
            'requires_data_files': requires_data_files,
            'related_explanation_chunk': f"{title}_explanation_0",
            'related_image_chunk': f"{title}_image_0" if 'image' in example_id.lower() else None
        }
        
        return CodeChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=code,
            content_type="code",
            metadata=metadata,
            source_type="example",
            original_id=example_id
        )


def main():
    """Test the code chunker"""
    print("=" * 80)
    print("CodeOnlyChunker - Test")
    print("=" * 80)
    
    # Test example with mixed imports (the problem case)
    test_example = {
        'id': 'TestExample',
        'title': 'TestExamplePythonicAPI',
        'code': '''import vtkmodules.all as vtk
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

def create_cylinder():
    """Create a cylinder"""
    # Create cylinder
    cylinder = vtk.vtkCylinderSource()  # Mixed usage!
    cylinder.SetRadius(1.0)
    cylinder.SetHeight(2.0)
    
    # Create mapper
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cylinder.GetOutputPort())
    
    # Create actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    
    return actor

def main():
    print("Creating visualization...")
    actor = create_cylinder()
    print("Done!")

if __name__ == "__main__":
    main()
''',
        'code_queries': [
            'How do I create a cylinder in VTK?',
            'Show me code to render a cylinder'
        ],
        'data_files': [],
        'data_download_info': []
    }
    
    # Run chunker
    chunker = CodeOnlyChunker()
    chunks = chunker.chunk_example(test_example)
    
    print(f"\nGenerated {len(chunks)} chunk(s)")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Tokens: ~{len(chunk.content) // 4}")
        print(f"Metadata:")
        print(f"  source_style: {chunk.metadata['source_style']}")
        print(f"  import_style: {chunk.metadata['import_style']}")
        print(f"  vtk_classes: {chunk.metadata['vtk_classes']}")
        print(f"  complexity: {chunk.metadata['complexity']}")
        print(f"  has_visualization: {chunk.metadata['has_visualization']}")
        print(f"\nCode preview (first 300 chars):")
        print(chunk.content[:300])
        print("...")
        print()
    
    print("=" * 80)
    print("✓ CodeOnlyChunker test complete")


if __name__ == '__main__':
    main()
