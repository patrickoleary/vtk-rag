#!/usr/bin/env python3
"""
Example Usage for TaskSpecificRetriever

Demonstrates various retrieval patterns with content-type filtering:
1. CODE retrieval (pythonic, self-contained)
2. EXPLANATION retrieval
3. API documentation lookup
4. Mixed retrieval
5. Configuration-based retrieval
6. Token comparison with old system
"""

from task_specific_retriever import (
    TaskSpecificRetriever,
    TaskType,
    RetrievalConfig
)


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def main():
    """Demonstrate task-specific retrieval patterns"""
    
    print_section("TaskSpecificRetriever - Example Usage")
    
    # Initialize retriever
    print("\nInitializing TaskSpecificRetriever...")
    retriever = TaskSpecificRetriever()
    
    # Example 1: CODE retrieval (pythonic, self-contained)
    print_section("Example 1: CODE Chunks (Pythonic, Self-Contained)")
    
    query = "How to create a cylinder in VTK?"
    print(f"\nQuery: '{query}'")
    
    code_results = retriever.retrieve_code(
        query,
        top_k=3,
        prefer_pythonic=True,
        prefer_self_contained=True
    )
    
    retriever.print_results_summary(code_results, "CODE Results")
    
    # Example 2: EXPLANATION retrieval
    print_section("Example 2: EXPLANATION Chunks")
    
    query = "What is a cylinder source?"
    print(f"\nQuery: '{query}'")
    
    explanation_results = retriever.retrieve_explanation(
        query,
        top_k=3
    )
    
    retriever.print_results_summary(explanation_results, "EXPLANATION Results")
    
    # Example 3: API documentation lookup
    print_section("Example 3: API Documentation (vtkActor)")
    
    query = "vtkActor methods"
    print(f"\nQuery: '{query}'")
    
    api_results = retriever.retrieve_api_doc(
        query,
        top_k=3,
        class_name="vtkActor"
    )
    
    retriever.print_results_summary(api_results, "API_DOC Results")
    
    # Example 4: IMAGE retrieval
    print_section("Example 4: IMAGE Chunks (Result Images)")
    
    query = "cylinder visualization output"
    print(f"\nQuery: '{query}'")
    
    image_results = retriever.retrieve_image(
        query,
        top_k=3,
        image_type="result"
    )
    
    retriever.print_results_summary(image_results, "IMAGE Results")
    print("\n💡 Image chunks are lightweight (~0 tokens) - only metadata with links to visual results")
    
    # Example 5: Filter by complexity
    print_section("Example 5: Simple Examples Only")
    
    query = "basic rendering example"
    print(f"\nQuery: '{query}'")
    
    simple_results = retriever.retrieve_code(
        query,
        top_k=3,
        complexity_level="simple",
        require_visualization=True
    )
    
    retriever.print_results_summary(simple_results, "SIMPLE CODE Results")
    
    # Example 6: Filter by VTK classes
    print_section("Example 6: Examples Using Specific VTK Classes")
    
    query = "rendering with vtkRenderer"
    print(f"\nQuery: '{query}'")
    
    class_results = retriever.retrieve_code(
        query,
        top_k=3,
        vtk_classes=["vtkRenderer", "vtkActor"]
    )
    
    retriever.print_results_summary(class_results, "VTK Class Filtered Results")
    
    # Example 7: Filter by category (explanations)
    print_section("Example 7: Explanations by Category")
    
    query = "geometric objects"
    print(f"\nQuery: '{query}'")
    
    category_results = retriever.retrieve_explanation(
        query,
        top_k=3,
        category="GeometricObjects"
    )
    
    retriever.print_results_summary(category_results, "Category Filtered Results")
    
    # Example 8: Mixed retrieval
    print_section("Example 8: Mixed Content Types")
    
    query = "cylinder example"
    print(f"\nQuery: '{query}'")
    
    mixed_results = retriever.retrieve_mixed(
        query,
        code_k=2,
        explanation_k=2,
        api_k=1,
        prefer_pythonic=True,
        prefer_self_contained=True
    )
    
    print(f"\nRetrieved:")
    print(f"  CODE: {len(mixed_results['code'])} chunks")
    print(f"  EXPLANATION: {len(mixed_results['explanation'])} chunks")
    print(f"  API_DOC: {len(mixed_results['api_doc'])} chunks")
    
    # Calculate total tokens
    all_chunks = (
        mixed_results['code'] +
        mixed_results['explanation'] +
        mixed_results['api_doc']
    )
    total_tokens = retriever.estimate_total_tokens(all_chunks)
    
    print(f"\n💡 Total tokens: {total_tokens}")
    print(f"   Old system: ~10,500 tokens (7 mixed chunks)")
    print(f"   Token reduction: ~{100 - (total_tokens / 10500 * 100):.0f}%")
    
    # Example 9: Configuration-based retrieval
    print_section("Example 9: Configuration-Based Retrieval")
    
    # CODE generation task
    print("\n8a. CODE Generation Task:")
    config = RetrievalConfig(
        task_type=TaskType.CODE_GENERATION,
        prefer_pythonic=True,
        prefer_self_contained=True,
        require_visualization=True
    )
    
    results = retriever.retrieve_with_config(
        "create a sphere",
        config,
        top_k=3
    )
    retriever.print_results_summary(results, "CODE Generation Task")
    
    # EXPLANATION task
    print("\n9b. EXPLANATION Task:")
    config = RetrievalConfig(
        task_type=TaskType.EXPLANATION,
        category="Rendering"
    )
    
    results = retriever.retrieve_with_config(
        "rendering concepts",
        config,
        top_k=2
    )
    retriever.print_results_summary(results, "EXPLANATION Task")
    
    # API lookup task
    print("\n9c. API Lookup Task:")
    config = RetrievalConfig(
        task_type=TaskType.API_LOOKUP,
        class_name="vtkPolyDataMapper"
    )
    
    results = retriever.retrieve_with_config(
        "mapper methods",
        config,
        top_k=2
    )
    retriever.print_results_summary(results, "API Lookup Task")
    
    # Example 10: Examples requiring data files
    print_section("Example 10: Examples WITH Data Files (for I/O tutorials)")
    
    query = "reading image files"
    print(f"\nQuery: '{query}'")
    
    # Don't filter out data files - we want I/O examples
    data_results = retriever.retrieve_code(
        query,
        top_k=3,
        prefer_pythonic=True,
        prefer_self_contained=False  # Allow data files
    )
    
    print(f"\nFound {len(data_results)} examples:")
    for i, result in enumerate(data_results, 1):
        requires_data = result.metadata.get('requires_data_files', False)
        data_files = result.metadata.get('data_files', [])
        print(f"\n{i}. {result.chunk_id}")
        print(f"   Requires Data: {requires_data}")
        if data_files:
            print(f"   Files Needed: {', '.join(data_files)}")
    
    # Example 11: Token comparison across scenarios
    print_section("Example 11: Token Comparison Summary")
    
    scenarios = [
        ("Simple code query", "create cylinder", 3, "CODE"),
        ("Explanation query", "cylinder geometry", 3, "EXPLANATION"),
        ("API lookup", "vtkActor methods", 3, "API"),
        ("Mixed query", "cylinder example", 5, "MIXED")
    ]
    
    print("\nScenario Comparison:")
    print(f"{'Scenario':<30} {'New System':<15} {'Old System':<15} {'Reduction'}")
    print("-" * 80)
    
    for scenario_name, query, top_k, rtype in scenarios:
        if rtype == "CODE":
            results = retriever.retrieve_code(query, top_k=top_k)
        elif rtype == "EXPLANATION":
            results = retriever.retrieve_explanation(query, top_k=top_k)
        elif rtype == "API":
            results = retriever.retrieve_api_doc(query, top_k=top_k)
        else:  # MIXED
            mixed = retriever.retrieve_mixed(query, code_k=2, explanation_k=2, api_k=1)
            results = mixed['code'] + mixed['explanation'] + mixed['api_doc']
        
        new_tokens = retriever.estimate_total_tokens(results)
        old_tokens = 10500  # 7 chunks × 1500 tokens
        reduction = 100 - (new_tokens / old_tokens * 100)
        
        print(f"{scenario_name:<30} {new_tokens:<15} {old_tokens:<15} {reduction:.0f}%")
    
    # Summary
    print_section("Summary")
    
    print("\n✅ Demonstrated:")
    print("  1. CODE retrieval (pythonic, self-contained)")
    print("  2. EXPLANATION retrieval")
    print("  3. API documentation lookup")
    print("  4. IMAGE retrieval (result images)")
    print("  5. Complexity filtering")
    print("  6. VTK class filtering")
    print("  7. Category filtering")
    print("  8. Mixed content-type retrieval")
    print("  9. Configuration-based retrieval")
    print("  10. Examples with/without data files")
    print("  11. Token comparison across scenarios")
    
    print("\n💡 Key Benefits:")
    print("  • 85-95% token reduction vs old system")
    print("  • Targeted retrieval by content_type")
    print("  • Automatic pythonic style preference")
    print("  • Self-contained examples by default")
    print("  • Flexible filtering by metadata")
    print("  • Configurable retrieval strategies")
    
    print("\n🔗 Next Steps:")
    print("  • Integrate with LLM generation pipeline")
    print("  • Add sequential thinking decomposition (Phase 10)")
    print("  • Build task-specific prompt templates")
    print("  • Add cross-encoder reranking (optional)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Qdrant is running: docker ps | grep qdrant")
        print("  2. Check index was built: python build-indexes/build_qdrant_index.py")
        print("  3. Verify connection: curl http://localhost:6333/")
        import traceback
        traceback.print_exc()
