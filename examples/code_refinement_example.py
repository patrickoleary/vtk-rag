#!/usr/bin/env python3
"""
Example: Code Refinement Feature

Demonstrates how to modify existing VTK code instead of regenerating from scratch.

Usage:
    python examples/code_refinement_example.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))

from sequential_pipeline import SequentialPipeline


def example_1_simple_modification():
    """Example 1: Simple property changes (no retrieval needed)"""
    print("=" * 80)
    print("EXAMPLE 1: Simple Modification - Change Color and Resolution")
    print("=" * 80)
    
    # Original code (basic cylinder)
    original_code = """from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor

cylinder_source = vtkCylinderSource()
cylinder_source.SetResolution(8)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder_source.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)

renderer = vtkRenderer()
renderer.AddActor(actor)

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
render_window.Render()
interactor.Start()
"""
    
    print("\nOriginal Code:")
    print("-" * 80)
    print(original_code)
    
    # Initialize pipeline
    pipeline = SequentialPipeline()
    
    # Refine the code (increase resolution and make it blue)
    print("\nModification Request: 'Increase resolution to 50 and make it blue'")
    print("-" * 80)
    
    result = pipeline.process_query(
        query="Increase resolution to 50 and make it blue",
        existing_code=original_code
    )
    
    print("\nModified Code:")
    print("-" * 80)
    print(result['code'])
    
    print("\nExplanation:")
    print("-" * 80)
    print(result['explanation'])
    
    print("\nModifications Applied:")
    for mod in result['modifications']:
        print(f"  {mod['step_number']}. {mod['modification']}")
    
    print("\nDiff:")
    print("-" * 80)
    print(result['diff'])
    
    print("\n" + "=" * 80 + "\n")


def example_2_adding_feature():
    """Example 2: Adding new feature (requires retrieval)"""
    print("=" * 80)
    print("EXAMPLE 2: Adding Feature - Add Texture Mapping")
    print("=" * 80)
    
    # Original code (basic sphere)
    original_code = """from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor

sphere = vtkSphereSource()
sphere.SetRadius(1.0)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)

renderer = vtkRenderer()
renderer.AddActor(actor)

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
render_window.Render()
interactor.Start()
"""
    
    print("\nOriginal Code:")
    print("-" * 80)
    print(original_code[:200] + "...")
    
    # Initialize pipeline
    pipeline = SequentialPipeline()
    
    # Refine the code (add texture)
    print("\nModification Request: 'Add texture mapping from earth.jpg'")
    print("-" * 80)
    
    result = pipeline.process_query(
        query="Add texture mapping from earth.jpg",
        existing_code=original_code
    )
    
    print("\nModifications Applied:")
    for mod in result['modifications']:
        print(f"  {mod['step_number']}. {mod['modification']}")
        print(f"     {mod['explanation'][:100]}...")
    
    print("\nNew Imports:")
    for imp in result.get('new_imports', []):
        print(f"  + {imp}")
    
    print("\nValidation:", "✓ Passed" if result.get('validation_passed') else "✗ Failed")
    print("Confidence:", result.get('confidence', 'unknown'))
    
    print("\n" + "=" * 80 + "\n")


def example_3_multiple_modifications():
    """Example 3: Multiple modifications in one request"""
    print("=" * 80)
    print("EXAMPLE 3: Multiple Modifications")
    print("=" * 80)
    
    # Original code (cone)
    original_code = """from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

cone = vtkConeSource()
mapper = vtkPolyDataMapper(input_connection=cone.GetOutputPort())
actor = vtkActor(mapper=mapper)
"""
    
    print("\nOriginal Code:")
    print("-" * 80)
    print(original_code)
    
    # Initialize pipeline
    pipeline = SequentialPipeline()
    
    # Multiple modifications
    print("\nModification Request: 'Increase height to 3, add rotation 45 degrees, and make it green'")
    print("-" * 80)
    
    result = pipeline.process_query(
        query="Increase height to 3, add rotation 45 degrees, and make it green",
        existing_code=original_code
    )
    
    print("\nModified Code:")
    print("-" * 80)
    print(result['code'])
    
    print("\nModifications Applied:")
    for i, mod in enumerate(result['modifications'], 1):
        print(f"\n{i}. {mod['modification']}")
        print(f"   {mod['explanation']}")
        if mod.get('code_changed'):
            print(f"   Changed: {mod['code_changed']}")
        if mod.get('code_added'):
            print(f"   Added: {mod['code_added']}")
    
    print("\n" + "=" * 80 + "\n")


def example_4_diff_style():
    """Example 4: Diff-style explanation"""
    print("=" * 80)
    print("EXAMPLE 4: Diff-Style Explanation")
    print("=" * 80)
    
    # Original code
    original_code = """from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

cube = vtkCubeSource()
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cube.GetOutputPort())
actor = vtkActor(mapper=mapper)
"""
    
    print("\nOriginal Code:")
    print("-" * 80)
    print(original_code)
    
    # Initialize pipeline
    pipeline = SequentialPipeline()
    
    # Refine with diff-style explanation
    print("\nModification Request: 'Double the size'")
    print("Explanation Style: diff")
    print("-" * 80)
    
    result = pipeline.process_query(
        query="Double the size",
        existing_code=original_code,
        explanation_style="diff"
    )
    
    print("\nExplanation (Diff Style):")
    print("-" * 80)
    print(result['explanation'])
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "VTK RAG Code Refinement Examples" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    try:
        # Run examples
        example_1_simple_modification()
        
        # Uncomment to run additional examples
        # example_2_adding_feature()
        # example_3_multiple_modifications()
        # example_4_diff_style()
        
        print("\n✓ All examples completed successfully!")
        print("\nNote: Uncomment other examples in the script to see more refinement patterns.")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
