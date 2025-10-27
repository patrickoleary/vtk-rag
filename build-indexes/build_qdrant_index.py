#!/usr/bin/env python3
"""
Build Qdrant hybrid index (Vector + BM25) from VTK chunked corpus

Updated for new chunk structure with content types and rich metadata
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall with: pip install qdrant-client sentence-transformers")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_chunks(file_path: Path) -> List[Dict]:
    """Load chunks from JSONL file"""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create Qdrant collection with vector and BM25 support"""
    import time
    
    try:
        # Delete existing collection
        client.delete_collection(collection_name)
        logger.info(f"✓ Deleted existing collection: {collection_name}")
        time.sleep(1)  # Wait for deletion to complete
    except Exception:
        logger.info(f"No existing collection to delete: {collection_name}")
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f"✓ Created collection: {collection_name}")
    
    # Wait for collection to be fully initialized
    time.sleep(2)
    
    # Enable BM25 full-text search on content field
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema="text"
        )
        logger.info(f"✓ Created BM25 index on content field")
    except Exception as e:
        logger.warning(f"Could not create BM25 index: {e}")
        logger.info("Continuing without BM25 index - vector search will still work")
    
    # Create indexes for filtering
    indexes_to_create = [
        ("content_type", "keyword"),  # Filter by code/explanation/api_doc/image
        ("source_type", "keyword"),   # Filter by example/test/api
        ("metadata.source_style", "keyword"),  # Filter pythonic vs basic
        ("metadata.import_style", "keyword"),  # Filter modular vs monolithic
        ("metadata.has_visualization", "bool"),
        ("metadata.has_data_io", "bool"),
        ("metadata.requires_data_files", "bool"),
        ("metadata.complexity", "keyword"),
    ]
    
    for field_name, field_schema in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            logger.info(f"✓ Created index on {field_name}")
        except Exception as e:
            logger.warning(f"Could not create index on {field_name}: {e}")
    
    logger.info(f"✓ Collection {collection_name} ready for indexing")


def index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 100,
    id_offset: int = 0
):
    """Index chunks with embeddings"""
    logger.info(f"Indexing {len(chunks)} chunks (starting at ID {id_offset})...")
    
    points = []
    
    for i, chunk in enumerate(chunks):
        content = chunk['content']
        
        # Generate embedding
        vector = model.encode(content)
        
        # Create point with full metadata
        point = PointStruct(
            id=id_offset + i,
            vector=vector.tolist(),
            payload={
                "content": content,
                "chunk_id": chunk['chunk_id'],
                "chunk_index": chunk['chunk_index'],
                "total_chunks": chunk['total_chunks'],
                "content_type": chunk['content_type'],
                "source_type": chunk['source_type'],
                "original_id": chunk['original_id'],
                "metadata": chunk['metadata']
            }
        )
        
        points.append(point)
        
        # Batch upload
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"  Indexed {i+1}/{len(chunks)} chunks")
            points = []
    
    # Upload remaining
    if points:
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"  Indexed {len(chunks)}/{len(chunks)} chunks")


def test_query(client: QdrantClient, collection_name: str, model: SentenceTransformer):
    """Test vector search query with content type filtering"""
    query = "How do I create a cylinder in VTK?"
    logger.info(f"\nTesting query: '{query}'")
    
    query_vector = model.encode(query)
    
    # Test 1: Vector search for CODE chunks only
    logger.info("\nTest 1: CODE chunks (pythonic style preferred)")
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "code"}},
            ],
            "should": [
                {"key": "metadata.source_style", "match": {"value": "pythonic"}}
            ]
        },
        limit=3
    )
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result.score:.4f}")
        print(f"   Chunk: {result.payload['chunk_id']}")
        print(f"   Style: {result.payload['metadata'].get('source_style', 'N/A')}")
        print(f"   Preview: {result.payload['content'][:150]}...")
    
    # Test 2: EXPLANATION chunks
    logger.info("\n\nTest 2: EXPLANATION chunks")
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        query_filter={
            "must": [
                {"key": "content_type", "match": {"value": "explanation"}},
            ]
        },
        limit=2
    )
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result.score:.4f}")
        print(f"   Chunk: {result.payload['chunk_id']}")
        print(f"   Preview: {result.payload['content'][:150]}...")
    
    logger.info(f"\n\n✅ Index is working! Query via Qdrant UI at http://localhost:6333/dashboard")


def main():
    parser = argparse.ArgumentParser(
        description='Build Qdrant hybrid index from VTK chunked corpus'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory containing chunked JSONL files'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:6333',
        help='Qdrant server URL'
    )
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model'
    )
    parser.add_argument(
        '--collection',
        default='vtk_docs',
        help='Collection name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for indexing'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip test query after indexing'
    )
    
    args = parser.parse_args()
    
    # Initialize Qdrant client
    logger.info(f"Connecting to Qdrant at {args.url}")
    client = QdrantClient(url=args.url)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    vector_size = model.get_sentence_embedding_dimension()
    logger.info(f"Vector size: {vector_size}")
    
    # Define chunk files
    chunk_files = [
        ('code_chunks.jsonl', 'CODE'),
        ('explanation_chunks.jsonl', 'EXPLANATION'),
        ('api_doc_chunks.jsonl', 'API_DOC'),
        ('image_chunks.jsonl', 'IMAGE'),
    ]
    
    # Create collection
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 8: REBUILD QDRANT INDEX")
    logger.info(f"{'='*60}\n")
    
    create_collection(client, args.collection, vector_size)
    
    # Index all chunk types
    total_chunks = 0
    id_offset = 0
    
    for filename, chunk_type in chunk_files:
        file_path = args.data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {chunk_type} chunks from {filename}")
        logger.info(f"{'='*60}")
        
        chunks = load_chunks(file_path)
        logger.info(f"Loaded {len(chunks)} {chunk_type} chunks")
        
        index_chunks(client, args.collection, chunks, model, args.batch_size, id_offset)
        id_offset += len(chunks)
        total_chunks += len(chunks)
        
        logger.info(f"✓ Completed {chunk_type} chunks")
    
    logger.info(f"\n{'='*60}")
    logger.info("INDEXING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total chunks indexed: {total_chunks:,}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Qdrant URL: {args.url}")
    
    # Test query
    if not args.skip_test:
        test_query(client, args.collection, model)
    
    logger.info("\n✅ Phase 8 Complete: Qdrant index rebuilt successfully!")
    logger.info(f"\nNext: Phase 9 - Build TaskSpecificRetriever")

if __name__ == '__main__':
    main()
