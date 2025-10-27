#!/usr/bin/env python3
"""
Example Usage for VTK Prompt Templates (Production System)

Demonstrates the JSON-based structured prompt system used in production:
1. Query Decomposition - Break complex queries into logical steps
2. Per-Step Generation - Generate code for each step with documentation
3. Structured JSON I/O - Clean data exchange with LLM

This shows how the sequential pipeline uses centralized prompts.
"""

import json
from prompt_templates import VTKPromptTemplate


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_json(data, label="JSON Structure"):
    """Pretty print JSON data"""
    print(f"\n{label}:")
    print(json.dumps(data, indent=2))


def example1_decomposition_instructions():
    """Example 1: Get Decomposition Instructions"""
    print_section("Example 1: Query Decomposition Instructions")
    
    print("\nThe decomposition prompt breaks complex queries into logical steps.")
    print("Used by: sequential_pipeline.py in decompose_query()")
    
    template = VTKPromptTemplate()
    instructions = template.get_decomposition_instructions()
    
    print_json(instructions, "Decomposition Instructions")
    
    print("\n" + "-" * 80)
    print("How it's used in production:")
    print("-" * 80)
    code_example = '''
    # In sequential_pipeline.py:
    from prompt_templates import VTKPromptTemplate
    
    template = VTKPromptTemplate()
    decomposition_input = {
        "query": user_query,
        "instructions": template.get_decomposition_instructions()
    }
    
    # Send to LLM for structured JSON response
    result = llm_client.generate_json(
        prompt_data=decomposition_input,
        schema_name="DecompositionOutput"
    )
    '''
    print(code_example)


def example2_generation_instructions():
    """Example 2: Get Generation Instructions"""
    print_section("Example 2: Per-Step Generation Instructions")
    
    print("\nThe generation prompt creates code for each step using retrieved docs.")
    print("Used by: sequential_pipeline.py in generate()")
    
    template = VTKPromptTemplate()
    instructions = template.get_generation_instructions()
    
    print_json(instructions, "Generation Instructions")
    
    print("\n" + "-" * 80)
    print("How it's used in production:")
    print("-" * 80)
    code_example = '''
    # In sequential_pipeline.py:
    template = VTKPromptTemplate()
    
    generation_input = {
        "original_query": query,
        "overall_understanding": understanding,
        "overall_plan": plan_dict,
        "current_step": current_step_dict,
        "previous_steps": previous_results,
        "documentation": retrieved_chunks,
        "instructions": template.get_generation_instructions()
    }
    
    # Send to LLM for structured JSON response
    step_response = llm_client.generate_json(
        prompt_data=generation_input,
        schema_name="GenerationOutput"
    )
    '''
    print(code_example)


def example3_complete_workflow():
    """Example 3: Complete Workflow Simulation"""
    print_section("Example 3: Complete Workflow Simulation")
    
    print("\nSimulating the full production pipeline flow...")
    
    template = VTKPromptTemplate()
    
    # Step 1: User query
    user_query = "How can I create a basic rendering of a polygonal cylinder in VTK?"
    print(f"\n📝 User Query: {user_query}")
    
    # Step 2: Build decomposition input
    print("\n🔧 Step 1: Build Decomposition Input")
    decomposition_input = {
        "query": user_query,
        "instructions": template.get_decomposition_instructions()
    }
    print(f"   ✓ Prepared decomposition input")
    print(f"   ✓ Instructions include: {list(decomposition_input['instructions'].keys())}")
    
    # Simulate LLM response
    print("\n🤖 LLM processes decomposition and returns JSON...")
    mock_steps = [
        {
            "step_number": 1,
            "description": "Create a polygonal cylinder geometry source",
            "search_query": "VTK vtkCylinderSource create polygon cylinder",
            "focus": "geometry"
        },
        {
            "step_number": 2,
            "description": "Create a mapper to convert geometry to graphics",
            "search_query": "VTK vtkPolyDataMapper map geometry",
            "focus": "rendering"
        }
    ]
    print(f"   ✓ LLM decomposed query into {len(mock_steps)} steps")
    
    # Step 3: For each step, build generation input
    print("\n🔧 Step 2: Build Generation Inputs for Each Step")
    for i, step in enumerate(mock_steps, 1):
        print(f"\n   Step {i}: {step['description']}")
        
        generation_input = {
            "original_query": user_query,
            "overall_understanding": "Create basic cylinder rendering",
            "overall_plan": {"total_steps": len(mock_steps), "current_step_number": i},
            "current_step": step,
            "previous_steps": [],
            "documentation": [],
            "instructions": template.get_generation_instructions()
        }
        
        print(f"      ✓ Prepared generation input with {len(generation_input['instructions']['requirements'])} requirements")
    
    print("\n✅ Complete workflow demonstration finished!")
    print("\nIn production, this flow is handled by:")
    print("   → llm-generation/sequential_pipeline.py")
    print("   → Uses: VTKPromptTemplate from grounding-prompting/")


def example4_integration():
    """Example 4: Integration with Sequential Pipeline"""
    print_section("Example 4: Integration Example")
    
    print("\nTo use these prompts in your own code:")
    print("-" * 80)
    
    integration_code = '''
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / 'grounding-prompting'))
sys.path.append(str(Path(__file__).parent / 'llm-generation'))

from prompt_templates import VTKPromptTemplate
from llm_client import LLMClient

# Initialize
template = VTKPromptTemplate()
llm_client = LLMClient()

# Decomposition
decomp_input = {
    "query": "Your query here",
    "instructions": template.get_decomposition_instructions()
}
result = llm_client.generate_json(decomp_input, "DecompositionOutput")

# Generation (for each step)
gen_input = {
    "original_query": "Your query",
    "overall_understanding": result["understanding"],
    "overall_plan": {...},
    "current_step": {...},
    "previous_steps": [...],
    "documentation": [...],
    "instructions": template.get_generation_instructions()
}
step_result = llm_client.generate_json(gen_input, "GenerationOutput")
    '''
    print(integration_code)


def main():
    """Run all examples"""
    print("\n" + "="* 80)
    print("VTK Prompt Templates - Production System Examples")
    print("=" * 80)
    print("\nThis demonstrates the JSON-based prompt system used by sequential_pipeline.py")
    print("All prompts are centralized in grounding-prompting/prompt_templates.py")
    
    example1_decomposition_instructions()
    example2_generation_instructions()
    example3_complete_workflow()
    example4_integration()
    
    print("\n" + "=" * 80)
    print("✅ All Examples Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. See: llm-generation/sequential_pipeline.py for production usage")
    print("  2. See: llm-generation/SCHEMAS.md for JSON schema documentation")
    print("  3. See: tests/grounding-prompting/ for unit tests")
    print("\n")


if __name__ == "__main__":
    main()
