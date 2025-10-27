#!/usr/bin/env python3
"""
Example: Using LLM Enrichment for Code Explanations

Demonstrates:
1. Processing a code response
2. Enriching with LLM-generated/improved explanations
3. Comparing original vs enriched
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'post-processing'))
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))

from json_response_processor import JSONResponseProcessor


def example_without_explanation():
    """Example: Code response with NO explanation - LLM generates one"""
    
    print("=" * 60)
    print("EXAMPLE 1: Generate Explanation for Code")
    print("=" * 60)
    
    # Simulated response from pipeline (no explanation)
    response = {
        "response_type": "answer",
        "content_type": "code",
        "code": """from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

cylinder = vtkCylinderSource()
cylinder.SetResolution(20)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)""",
        "explanation": "",  # Empty explanation
        "confidence": "high",
        "citations": []
    }
    
    # Simulated documentation chunks
    documentation_chunks = [
        {"content": "vtkCylinderSource creates a 3D cylinder polydata. Use SetResolution to control smoothness."},
        {"content": "vtkPolyDataMapper maps polydata to graphics primitives for rendering."},
        {"content": "vtkActor represents an object in the rendering scene."}
    ]
    
    # Process with enrichment
    processor = JSONResponseProcessor()
    
    print("\n📝 Original Response:")
    print(f"  Explanation: '{response['explanation']}'")
    print(f"  Code length: {len(response['code'])} chars")
    
    # Enrich (would call LLM)
    print("\n🤖 Enriching with LLM...")
    print("   (In real usage, this would call the LLM to generate explanation)")
    print("   For this demo, we'll show what the enriched response structure looks like")
    
    # Show expected enriched structure
    print("\n✨ Enriched Response Would Contain:")
    print("  - improved_explanation: Detailed explanation of the code")
    print("  - key_points: ['Creates cylinder', 'Maps to graphics', 'Adds to scene']")
    print("  - vtk_classes_explained: [{name: 'vtkCylinderSource', purpose: '...'}]")
    print("  - enrichment_confidence: 'high'")


def example_with_poor_explanation():
    """Example: Code response with poor explanation - LLM improves it"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Improve Existing Poor Explanation")
    print("=" * 60)
    
    # Simulated response with minimal explanation
    response = {
        "response_type": "answer",
        "content_type": "code",
        "code": """from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

reader = vtkSTLReader()
reader.SetFileName('model.stl')

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 0, 0)""",
        "explanation": "Reads STL file and renders it.",  # Minimal explanation
        "confidence": "medium",
        "citations": []
    }
    
    documentation_chunks = [
        {"content": "vtkSTLReader reads STL files (STereoLithography format), commonly used for 3D printing."},
        {"content": "GetProperty() returns vtkProperty object for setting visual properties like color, opacity."},
        {"content": "SetColor(r, g, b) sets RGB color values between 0 and 1."}
    ]
    
    processor = JSONResponseProcessor()
    
    print("\n📝 Original Response:")
    print(f"  Explanation: '{response['explanation']}'")
    print(f"  Quality: Poor (too brief)")
    
    print("\n🤖 Improving with LLM...")
    print("   (In real usage, this would call the LLM to improve explanation)")
    
    print("\n✨ Improved Explanation Would Include:")
    print("  - What STL format is and why it's used")
    print("  - Explanation of the VTK pipeline (reader → mapper → actor)")
    print("  - Details about the property settings (red color)")
    print("  - Best practices for file paths")


def example_non_code_passthrough():
    """Example: Non-code response - passes through unchanged"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Non-Code Response (Pass Through)")
    print("=" * 60)
    
    # API response
    response = {
        "response_type": "answer",
        "content_type": "api",
        "explanation": "SetMapper assigns a vtkMapper to a vtkActor for rendering.",
        "confidence": "high",
        "citations": []
    }
    
    processor = JSONResponseProcessor()
    
    print("\n📝 API Response:")
    print(f"  Content Type: {response['content_type']}")
    print(f"  Explanation: {response['explanation']}")
    
    print("\n➡️  Passes through unchanged (not code)")
    print("   Future: May enrich API responses with usage examples")


def usage_in_pipeline():
    """Show how to integrate enrichment into pipeline"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION: Using in Pipeline")
    print("=" * 60)
    
    print("""
# In your pipeline code:

from json_response_processor import JSONResponseProcessor

# After getting response from query handler
response = pipeline.process_query(query)

# Create processor
processor = JSONResponseProcessor()

# Option 1: Process without enrichment (fast)
enriched = processor.process(response)
print(enriched.vtk_classes)
print(enriched.metadata)

# Option 2: Process WITH LLM enrichment (slower, better explanations)
enriched_response = processor.enrich_with_llm(
    response=response,
    documentation_chunks=retrieved_chunks,
    llm_client=your_llm_client  # Optional
)

# Now has improved explanation
print(enriched_response['explanation'])  # Enhanced!
print(enriched_response['_enrichment']['key_points'])
    """)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VTK RAG: Code Explanation Enrichment Examples")
    print("=" * 60)
    
    example_without_explanation()
    example_with_poor_explanation()
    example_non_code_passthrough()
    usage_in_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)
    print("""
Key Features:
✅ Generates explanations when missing
✅ Improves poor/minimal explanations  
✅ Uses documentation context for accuracy
✅ Passes through non-code responses
✅ Optional - only call when needed
✅ Preserves original in metadata
    """)
