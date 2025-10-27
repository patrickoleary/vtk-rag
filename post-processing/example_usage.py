#!/usr/bin/env python3
"""
Example Usage: Post-Processing

Demonstrates how to use the response parser and enricher to:
1. Parse LLM responses into structured components
2. Handle interactive flows (clarifying questions)
3. Extract code, explanations, citations separately
4. Enrich CODE responses with generated explanations
5. Validate responses
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / 'llm-generation'))

from response_parser import (
    ResponseParser,
    ResponseType,
    ConfidenceLevel,
    ParsedResponse
)
from response_enricher import ResponseEnricher, EnrichedResponse


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def example1_complete_answer():
    """Example 1: Parse a complete answer with code"""
    print_section("Example 1: Complete Answer with Code and Citations")
    
    response_text = """To create a cylinder with radius 5 and height 10, use vtkCylinderSource [1]:

```python
from vtkmodules.vtkFiltersSources import vtkCylinderSource

cylinder = vtkCylinderSource()
cylinder.SetRadius(5.0)
cylinder.SetHeight(10.0)
cylinder.Update()
```

The vtkCylinderSource class creates a polygonal cylinder centered at the origin [1]. 
By default, the cylinder is oriented along the Y-axis. You can use SetCenter() to 
position it and SetResolution() to control the polygon count [1].

For rendering, connect to vtkPolyDataMapper [2]:

```python
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)
```

This approach is shown in the VTK examples [2].
"""
    
    parser = ResponseParser()
    parsed = parser.parse(response_text)
    
    print(f"\n📊 Response Analysis:")
    print(f"  Type: {parsed.response_type.value}")
    print(f"  Has code: {parsed.has_code}")
    print(f"  Has citations: {parsed.has_citations}")
    print(f"  Confidence: {parsed.confidence.value}")
    
    print(f"\n💻 Code Blocks: {len(parsed.code_blocks)}")
    for i, block in enumerate(parsed.code_blocks, 1):
        print(f"\n  Block {i}:")
        print(f"    Lines: {len(block.code.splitlines())}")
        print(f"    Has imports: {block.has_imports}")
        print(f"    VTK classes: {', '.join(block.vtk_classes)}")
    
    print(f"\n📝 Explanations: {len(parsed.explanations)}")
    for i, explanation in enumerate(parsed.explanations, 1):
        preview = explanation[:100] + "..." if len(explanation) > 100 else explanation
        print(f"  {i}. {preview}")
    
    print(f"\n📚 Citations: {[str(c) for c in parsed.citations]}")
    
    print(f"\n🔧 API References: {len(parsed.api_references)}")
    for api in parsed.api_references:
        methods_str = ', '.join(api.methods) if api.methods else "none"
        print(f"  • {api.class_name}: {methods_str}")
    
    print(f"\n✓ Can extract main code for execution")
    print(f"✓ Can display explanations separately")
    print(f"✓ Can link citations to sources")


def example2_clarifying_question():
    """Example 2: Parse a clarifying question (interactive flow)"""
    print_section("Example 2: Clarifying Question (Interactive Flow)")
    
    response_text = """Could you please clarify which type of visualization you need?

I can help you with:
1. Basic cylinder rendering (simple visualization)
2. Cylinder with custom colors and properties
3. Multiple cylinders in a scene
4. Animated rotating cylinder

Which option would you prefer?"""
    
    parser = ResponseParser()
    parsed = parser.parse(response_text)
    
    print(f"\n📊 Response Analysis:")
    print(f"  Type: {parsed.response_type.value}")
    print(f"  Interactive: {parsed.is_interactive}")
    
    print(f"\n❓ Question:")
    print(f"  {parsed.question[:100]}...")
    
    print(f"\n📋 Options: {len(parsed.options)}")
    for i, option in enumerate(parsed.options, 1):
        print(f"  {i}. {option}")
    
    print(f"\n💡 Next Step:")
    print(f"  Present options to user → User selects → Re-query with clarification")


def example3_refusal():
    """Example 3: Parse a refusal response"""
    print_section("Example 3: Refusal Response")
    
    response_text = """I don't have enough information in the provided VTK documentation to answer this question accurately. The context discusses cylinders and spheres, but your question is about volume rendering of medical images. Please consult the official VTK documentation or ask a more specific question about the geometric primitives covered in the provided examples."""
    
    parser = ResponseParser()
    parsed = parser.parse(response_text)
    
    print(f"\n📊 Response Analysis:")
    print(f"  Type: {parsed.response_type.value}")
    
    print(f"\n⚠️  Refusal Reason:")
    print(f"  {parsed.refusal_reason}")
    
    print(f"\n💡 Next Step:")
    print(f"  Inform user → Suggest alternative query or broader search")


def example4_integrated_pipeline():
    """Example 4: Integration with full RAG pipeline"""
    print_section("Example 4: Integrated Pipeline")
    
    print("\nComplete RAG workflow with post-processing:\n")
    print("""
# 1. Query → 2. Retrieve → 3. Ground → 4. Generate → 5. Post-Process

from response_parser import ResponseParser

# ... (retrieval and generation steps)

# After LLM generation:
raw_response = generator.generate(prompt, metadata)
parser = ResponseParser()
parsed = parser.parse(raw_response.answer, metadata)

# Now you can:
if parsed.response_type == ResponseType.CLARIFYING_QUESTION:
    # Interactive flow - present options to user
    display_options(parsed.options)
    user_choice = get_user_input()
    # Re-query with clarification
    new_query = f"{original_query} (specifically: {user_choice})"
    # ... restart pipeline
    
elif parsed.response_type == ResponseType.ANSWER:
    # Display structured answer
    if parsed.code_blocks:
        display_code(parsed.get_main_code())
    
    for explanation in parsed.explanations:
        display_text(explanation)
    
    if parsed.citations:
        display_sources(parsed.citations, metadata)
    
    display_confidence(parsed.confidence)
    
elif parsed.response_type == ResponseType.REFUSAL:
    # Handle refusal gracefully
    display_refusal(parsed.refusal_reason)
    suggest_alternatives()
""")


def example5_component_extraction():
    """Example 5: Extract specific components for different uses"""
    print_section("Example 5: Component Extraction for Different Uses")
    
    response_text = """Here's how to create a sphere using vtkSphereSource [1]:

```python
from vtkmodules.vtkFiltersSources import vtkSphereSource

sphere = vtkSphereSource()
sphere.SetCenter(0.0, 0.0, 0.0)
sphere.SetRadius(2.0)
sphere.Update()
```

The sphere is created with a default resolution of 8 divisions [1]."""
    
    parser = ResponseParser()
    parsed = parser.parse(response_text)
    
    print("\n📤 Use Case 1: Extract for Code Execution")
    print("-" * 60)
    main_code = parsed.get_main_code()
    print(f"Extracted {len(main_code.splitlines())} lines of runnable code")
    
    print("\n📤 Use Case 2: Extract for Documentation")
    print("-" * 60)
    print(f"API classes documented: {parsed.get_all_vtk_classes()}")
    
    print("\n📤 Use Case 3: Extract for Citation Validation")
    print("-" * 60)
    print(f"Citations to validate: {[c.number for c in parsed.citations]}")
    
    print("\n📤 Use Case 4: Extract for UI Rendering")
    print("-" * 60)
    print(f"Code panels: {len(parsed.code_blocks)}")
    print(f"Text panels: {len(parsed.explanations)}")
    print(f"Citation badges: {len(parsed.citations)}")


def example6_data_files_and_baselines():
    """Example 6: Extract data files and baseline images"""
    print_section("Example 6: Data Files & Baseline Images")
    
    response_text = """To load and process the mesh file, use vtkSTLReader [1]:

```python
from vtkmodules.vtkIOGeometry import vtkSTLReader

reader = vtkSTLReader()
reader.SetFileName('headMesh.stl')
reader.Update()
```

This example requires the headMesh.stl data file [1]."""
    
    # Mock metadata with data files and baseline
    metadata = {
        'chunk_details': [
            {
                'position': 1,
                'chunk_id': 'MeshProcessing_0',
                'source_type': 'example',
                'title': 'STL Mesh Processing',
                'has_image': True,
                'image_url': 'https://examples.vtk.org/images/MeshProcessing.png',
                'data_download_info': [
                    {
                        'filename': 'headMesh.stl',
                        'download_urls': [
                            {
                                'type': 'vtk_examples_gitlab',
                                'url': 'https://gitlab.kitware.com/vtk/vtk-examples/-/raw/master/src/Testing/Data/headMesh.stl',
                                'method': 'direct_download'
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    parser = ResponseParser()
    parsed = parser.parse(response_text, metadata)
    
    print("\n📦 Data Files:")
    if parsed.has_data_files():
        for df in parsed.data_files:
            print(f"  • {df.filename}")
            if df.download_urls:
                print(f"    Download: {df.download_urls[0]['url']}")
    
    print("\n🖼️  Baseline Images:")
    if parsed.has_baseline_images():
        baseline = parsed.get_baseline_image()
        if baseline:
            print(f"  • URL: {baseline.image_url}")
            print(f"    Source: {baseline.source_type}")
    
    print("\n📝 Formatted Data Section:")
    data_section = parsed.format_data_section()
    if data_section:
        print(data_section)
    
    print("\n📝 Formatted Baseline Section:")
    baseline_section = parsed.format_baseline_section()
    if baseline_section:
        print(baseline_section)


def example7_code_enrichment():
    """Example 7: Enrich CODE response with generated explanation"""
    print_section("Example 7: CODE Response Enrichment (Requires API Key)")
    
    response_text = """To create a cylinder with radius 5, use vtkCylinderSource [1]:

```python
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

cylinder = vtkCylinderSource()
cylinder.SetRadius(5.0)
cylinder.SetHeight(10.0)
cylinder.Update()

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)
```

This creates a cylinder with the specified dimensions [1]."""
    
    metadata = {
        'query': 'Create a cylinder',
        'chunks_used': 1,
        'chunk_details': [
            {
                'position': 1,
                'chunk_id': 'CylinderExample_code_0',
                'source_type': 'example',
                'content_type': 'code',
                'score': 0.95
            }
        ]
    }
    
    # Step 1: Parse
    parser = ResponseParser()
    parsed = parser.parse(response_text, metadata)
    
    print(f"\n📊 Parsed Response:")
    print(f"  Has code: {parsed.has_code}")
    print(f"  VTK classes: {', '.join(parsed.get_all_vtk_classes())}")
    
    # Step 2: Enrich (for CODE responses)
    print(f"\n🔄 Enriching CODE response...")
    print(f"  (This makes a 2nd LLM call to explain the code)")
    
    try:
        enricher = ResponseEnricher()
        enriched = enricher.enrich(
            parsed=parsed,
            content_type='code',
            metadata=metadata,
            generate_explanation=True  # Set to False to skip (no API call)
        )
        
        if enriched.has_enrichment:
            print(f"\n✅ Enrichment successful!")
            
            # Get all display sections
            sections = enriched.get_display_sections()
            
            print(f"\n📦 Available Sections:")
            for key in sections.keys():
                if sections[key]:
                    print(f"  ✓ {key}")
            
            if sections.get('explanation'):
                print(f"\n📝 Generated Explanation:")
                print(f"  {sections['explanation'][:200]}...")
            
            print(f"\n💡 For API/EXPLANATION/IMAGE queries:")
            print(f"  No enrichment needed - parser output is sufficient")
        else:
            print(f"\n⚠️  Enrichment skipped (no API key or error)")
            print(f"  Parsed response still available for display")
    
    except Exception as e:
        print(f"\n⚠️  Enrichment failed: {e}")
        print(f"  (This is normal if no .env file with API key)")
        print(f"  Parsed response still usable without enrichment")


def main():
    print("=" * 80)
    print("Post-Processing - Example Usage")
    print("=" * 80)
    
    try:
        example1_complete_answer()
        example2_clarifying_question()
        example3_refusal()
        example4_integrated_pipeline()
        example5_component_extraction()
        example6_data_files_and_baselines()
        example7_code_enrichment()
        
        # Summary
        print_section("Summary")
        print("\n✅ Post-processing capabilities:")
        print("  • Parse response types (answer, question, refusal)")
        print("  • Extract code blocks with VTK class detection")
        print("  • Separate explanations from code")
        print("  • Link citations to source chunks")
        print("  • Detect confidence levels")
        print("  • Enable interactive flows")
        print("  • Extract data files with download URLs")
        print("  • Format baseline images and expected outputs")
        print("  • Enrich CODE responses with generated explanations (2nd LLM pass)")
        print("  • Structured output for UI rendering")
        
        print("\n📌 Integration points:")
        print("  • After LLM generation (stage 5)")
        print("  • Before UI display")
        print("  • Before evaluation")
        print("  • For interactive chat loops")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
