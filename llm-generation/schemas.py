#!/usr/bin/env python3
"""JSON Schema Definitions for LLM-Pipeline Communication"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json

# Decomposition schemas
@dataclass
class DecompositionInput:
    query: str
    instructions: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class Step:
    step_number: int
    description: str
    search_query: str
    focus: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DecompositionOutput:
    understanding: str
    requires_visualization: bool
    libraries_needed: List[str]
    data_files: List[str]
    steps: List[Dict]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecompositionOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_steps(self) -> List[Step]:
        return [Step(**s) for s in self.steps]

# Generation schemas
@dataclass
class DocumentationChunk:
    index: int
    chunk_id: str
    content_type: str
    content: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class PreviousStepResult:
    step_number: int
    understanding: str
    imports: List[str]
    code: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CurrentStepInfo:
    step_number: int
    description: str
    focus: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class OverallPlan:
    total_steps: int
    current_step_number: int
    steps: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class GenerationInput:
    original_query: str
    overall_understanding: str
    overall_plan: Dict
    current_step: Dict
    previous_steps: List[Dict]
    documentation: List[Dict]
    instructions: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class GenerationOutput:
    step_number: int
    understanding: str
    imports: List[str]
    code: str
    citations: List[int]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GenerationOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

# Validation schemas
@dataclass
class ValidationError:
    type: str
    message: str
    line: int
    context: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ValidationFix:
    error_type: str
    fix: str
    line: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ValidationInput:
    task: str
    original_query: str
    generated_code: str
    validation_errors: List[Dict]
    instructions: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ValidationOutput:
    fixed_code: str
    changes_made: List[Dict]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValidationOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

# Final result schemas
@dataclass
class FinalCode:
    imports: str
    body: str
    complete: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Explanation:
    overview: str
    imports: List[str]
    steps: List[Dict]
    formatted: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class StepResult:
    step_number: int
    description: str
    understanding: str
    imports: List[str]
    code: str
    citations: List[int]
    chunks_used: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class FinalResult:
    query: str
    understanding: str
    requires_visualization: bool
    libraries_needed: List[str]
    data_files: List[str]
    steps: List[Dict]
    final_code: Dict
    explanation: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

# New query type schemas (Phase 1-3)
@dataclass
class APILookupOutput:
    response_type: str
    content_type: str
    explanation: str
    confidence: str
    citations: List[Dict]
    usage_example: str = ""
    parameters: List[Dict] = None
    return_value: str = ""
    related_methods: List[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'APILookupOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ExplanationOutput:
    response_type: str
    content_type: str
    explanation: str
    confidence: str
    citations: List[Dict]
    key_concepts: List[Dict] = None
    examples: List[str] = None
    related_concepts: List[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExplanationOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DataToCodeOutput:
    response_type: str
    content_type: str
    code: str
    explanation: str
    confidence: str
    citations: List[Dict]
    data_analysis: str = ""
    suggested_techniques: List[str] = None
    alternative_approaches: List[Dict] = None
    vtk_classes_used: List[str] = None
    data_files_used: List[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataToCodeOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CodeToDataOutput:
    response_type: str
    content_type: str
    explanation: str
    confidence: str
    citations: List[Dict]
    code_requirements: str = ""
    data_files: List[Dict] = None
    vtk_classes_used: List[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeToDataOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ExplanationEnrichmentOutput:
    """Enriched or generated explanation for code"""
    improved_explanation: str
    confidence: str
    citations: List[Dict]
    key_points: List[str] = None
    vtk_classes_explained: List[Dict] = None  # [{name: str, purpose: str}]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExplanationEnrichmentOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

# Validation functions
def validate_decomposition_output(data: Dict) -> bool:
    required = ['understanding', 'requires_visualization', 'libraries_needed', 'data_files', 'steps']
    return all(f in data for f in required)

def validate_generation_output(data: Dict) -> bool:
    required = ['step_number', 'understanding', 'imports', 'code', 'citations']
    return all(f in data for f in required)

def validate_validation_output(data: Dict) -> bool:
    required = ['fixed_code', 'changes_made']
    return all(f in data for f in required)

def validate_api_lookup_output(data: Dict) -> bool:
    required = ['response_type', 'content_type', 'explanation', 'confidence', 'citations']
    return all(f in data for f in required)

def validate_explanation_output(data: Dict) -> bool:
    required = ['response_type', 'content_type', 'explanation', 'confidence', 'citations']
    return all(f in data for f in required)

def validate_data_to_code_output(data: Dict) -> bool:
    required = ['response_type', 'content_type', 'code', 'explanation', 'confidence', 'citations']
    return all(f in data for f in required)

def validate_code_to_data_output(data: Dict) -> bool:
    required = ['response_type', 'content_type', 'explanation', 'confidence', 'citations']
    return all(f in data for f in required)

def validate_explanation_enrichment_output(data: Dict) -> bool:
    required = ['improved_explanation', 'confidence', 'citations']
    return all(f in data for f in required)

# ============ CODE REFINEMENT SCHEMAS ============

@dataclass
class ModificationDecompositionOutput:
    """Decomposition of modification request into steps"""
    understanding: str  # What the user wants to change
    modification_steps: List[Dict]  # List of {step_number, description, requires_retrieval}
    preserved_elements: List[str]  # What should NOT be changed
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModificationDecompositionOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CodeModificationOutput:
    """Result of code modification"""
    modifications: List[Dict]  # List of {step_number, modification, explanation, code_changed, code_added, variable_affected}
    updated_code: str  # Complete updated code
    new_imports: List[str]  # New imports added (if any)
    preserved_structure: bool  # Whether original structure was preserved
    diff_summary: str  # Human-readable summary of changes
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeModificationOutput':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CodeRefinementResult:
    """Complete refinement result"""
    response_type: str  # "answer"
    content_type: str  # "code_refinement"
    query: str  # Original modification request
    original_code: str  # Original code
    code: str  # Updated code (for consistency with generation)
    explanation: str  # Explanation of modifications
    modifications: List[Dict]  # Detailed modification steps
    new_imports: List[str]  # New imports added
    citations: List[Dict]  # Documentation citations
    chunk_ids_used: List[str]  # Retrieved chunk IDs
    confidence: str  # "high", "medium", "low"
    diff: str  # Diff-style changes (optional)
    
    def to_dict(self) -> Dict:
        return asdict(self)

def validate_modification_decomposition_output(data: Dict) -> bool:
    required = ['understanding', 'modification_steps', 'preserved_elements']
    return all(f in data for f in required)

def validate_code_modification_output(data: Dict) -> bool:
    required = ['modifications', 'updated_code']
    return all(f in data for f in required)
