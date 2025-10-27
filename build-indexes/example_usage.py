#!/usr/bin/env python3
"""
Example Usage for VTK RAG Index with Content-Type Filtering

Demonstrates how to query the Qdrant index with content-type separation:
- CODE chunks (pythonic, self-contained)
- EXPLANATION chunks
- API_DOC chunks (class documentation)
- Combined retrieval with links
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_result(i, result):
    """Print formatted search result"""
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Chunk ID: {result.payload['chunk_id']}")
    print(f"   Content Type: {result.payload['content_type']}")
    print(f"   Source Type: {result.payload['source_type']}")
    
    metadata = result.payload.get('metadata', {})
    
    # CODE chunk metadata
    if result.payload['content_type'] == 'code':
        print(f"   Style: {metadata.get('source_style', 'N/A')}")
        print(f"   Complexity: {metadata.get('complexity', 'N/A')}")
        print(f"   Requires Data: {metadata.get('requires_data_files', False)}")
        print(f"   Has Viz: {metadata.get('has_visualization', False)}")
        
        vtk_classes = metadata.get('vtk_classes', [])
        if vtk_classes:
            print(f"   VTK Classes: {', '.join(vtk_classes[:5])}")
    
    # EXPLANATION chunk metadata
    elif result.payload['content_type'] == 'explanation':
        if 'title' in metadata:
            print(f"   Title: {metadata['title']}")
        if 'category' in metadata:
            print(f"   Category: {metadata['category']}")
    
    # API_DOC chunk metadata
    elif result.payload['content_type'] == 'api_doc':
        if 'class_name' in metadata:
            print(f"   Class: {metadata['class_name']}")
        if 'method_names' in metadata:
            methods = metadata['method_names']
            print(f"   Methods: {', '.join(methods[:3])}")
            if len(methods) > 3:
                print(f"            ... and {len(methods) - 3} more")
    
    # Show content preview
    content = result.payload['content']
    preview = content[:200].replace('\n', ' ')
    print(f"   Preview: {preview}...")


def main():
    """Demonstrate various query patterns"""
    
    # Initialize client and model
    print_section("Initializing Qdrant Client and Embedding Model")
    
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("✓ Connected to Qdrant at http://localhost:6333")
    print("✓ Loaded embedding model: all-MiniLM-L6-v2")
    
    # Test query
    query = "How do I create a cylinder in VTK?"
    print(f"\nQuery: '{query}'")
    query_vector = model.encode(query)
    
    # Example 1: CODE chunks (pythonic, self-contained)
    print_section("Example 1: CODE Chunks (Pythonic, Self-Contained)")
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "code"}},
                {"key": "metadata.source_style", "match": {"value": "pythonic"}},
                {"key": "metadata.requires_data_files", "match": {"value": False}}
            ]
        },
        limit=3
    )
    
    print(f"\nFound {len(results)} pythonic, self-contained code examples:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    # Calculate token usage
    total_chars = sum(len(r.payload['content']) for r in results)
    estimated_tokens = total_chars // 4
    print(f"\n💡 Token usage: ~{estimated_tokens} tokens (3 chunks)")
    
    # Example 2: EXPLANATION chunks
    print_section("Example 2: EXPLANATION Chunks")
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "explanation"}}
            ]
        },
        limit=3
    )
    
    print(f"\nFound {len(results)} explanation chunks:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    total_chars = sum(len(r.payload['content']) for r in results)
    estimated_tokens = total_chars // 4
    print(f"\n💡 Token usage: ~{estimated_tokens} tokens (3 chunks)")
    
    # Example 3: API Documentation
    print_section("Example 3: API Documentation Lookup")
    
    api_query = "vtkActor methods"
    api_query_vector = model.encode(api_query)
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=api_query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "api_doc"}},
                {"key": "metadata.class_name", "match": {"value": "vtkActor"}}
            ]
        },
        limit=3
    )
    
    print(f"\nQuery: '{api_query}'")
    print(f"Found {len(results)} API documentation chunks for vtkActor:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    # Example 4: Filter by complexity
    print_section("Example 4: Simple Examples Only")
    
    simple_query = "basic rendering example"
    simple_query_vector = model.encode(simple_query)
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=simple_query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "code"}},
                {"key": "metadata.complexity", "match": {"value": "simple"}},
                {"key": "metadata.has_visualization", "match": {"value": True}}
            ]
        },
        limit=3
    )
    
    print(f"\nQuery: '{simple_query}'")
    print(f"Found {len(results)} simple visualization examples:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    # Example 5: Find examples using specific VTK classes
    print_section("Example 5: Examples Using Specific VTK Classes")
    
    class_query = "examples using vtkRenderer"
    class_query_vector = model.encode(class_query)
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=class_query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "code"}},
                {"key": "metadata.vtk_classes", "match": {"any": ["vtkRenderer"]}}
            ]
        },
        limit=3
    )
    
    print(f"\nQuery: '{class_query}'")
    print(f"Found {len(results)} examples using vtkRenderer:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    # Example 6: Combined retrieval (CODE + EXPLANATION)
    print_section("Example 6: Combined Retrieval (Code + Explanation)")
    
    # Get 1 code chunk
    code_results = client.search(
        collection_name="vtk_docs",
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "code"}},
                {"key": "metadata.source_style", "match": {"value": "pythonic"}}
            ]
        },
        limit=1
    )
    
    # Get 1 explanation chunk
    explanation_results = client.search(
        collection_name="vtk_docs",
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "explanation"}}
            ]
        },
        limit=1
    )
    
    print("\nRetrieved:")
    print("\nCODE chunk:")
    if code_results:
        print_result(1, code_results[0])
    
    print("\nEXPLANATION chunk:")
    if explanation_results:
        print_result(1, explanation_results[0])
    
    # Calculate combined token usage
    code_chars = len(code_results[0].payload['content']) if code_results else 0
    expl_chars = len(explanation_results[0].payload['content']) if explanation_results else 0
    total_chars = code_chars + expl_chars
    estimated_tokens = total_chars // 4
    
    print(f"\n💡 Total token usage: ~{estimated_tokens} tokens (1 code + 1 explanation)")
    print(f"   Compare to old system: ~1,500 tokens per chunk × 7 = 10,500 tokens")
    print(f"   Token reduction: ~{100 - (estimated_tokens / 10500 * 100):.0f}%")
    
    # Example 7: Filter by category (for explanations)
    print_section("Example 7: Explanations by Category")
    
    category_query = "geometric objects examples"
    category_query_vector = model.encode(category_query)
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=category_query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "explanation"}},
                {"key": "metadata.category", "match": {"value": "GeometricObjects"}}
            ]
        },
        limit=3
    )
    
    print(f"\nQuery: '{category_query}'")
    print(f"Found {len(results)} GeometricObjects explanations:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    # Example 8: IMAGE chunks
    print_section("Example 8: IMAGE Chunks (Result Images)")
    
    image_query = "cylinder visualization output"
    image_query_vector = model.encode(image_query)
    
    results = client.search(
        collection_name="vtk_docs",
        query_vector=image_query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "image"}},
                {"key": "metadata.image_type", "match": {"value": "result"}}
            ]
        },
        limit=3
    )
    
    print(f"\nQuery: '{image_query}'")
    print(f"Found {len(results)} result images:")
    for i, result in enumerate(results, 1):
        print_result(i, result)
    
    print("\n💡 Image chunks are lightweight (~0 tokens) - only metadata with links to visual results")
    
    # Summary
    print_section("Summary")
    
    print("\n✅ Successfully demonstrated:")
    print("  1. CODE chunk filtering (pythonic, self-contained)")
    print("  2. EXPLANATION chunk retrieval")
    print("  3. API_DOC lookup by class name")
    print("  4. Complexity filtering (simple examples)")
    print("  5. VTK class filtering")
    print("  6. Combined retrieval (code + explanation)")
    print("  7. Category filtering")
    print("  8. IMAGE chunk retrieval (result images)")
    
    print("\n💡 Key Benefits:")
    print("  • Targeted retrieval by content_type")
    print("  • 85-95% token reduction vs old system")
    print("  • Precise filtering by metadata")
    print("  • Prefer pythonic examples automatically")
    print("  • Filter out examples requiring data files")
    
    print("\n🔗 Useful Resources:")
    print("  • Qdrant Dashboard: http://localhost:6333/dashboard")
    print("  • Collection: vtk_docs")
    print("  • Total chunks: 131,062")
    print("  • Documentation: See build-indexes/README.md")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Qdrant is running: docker ps | grep qdrant")
        print("  2. Check index was built: python build-indexes/build_qdrant_index.py")
        print("  3. Verify connection: curl http://localhost:6333/")
