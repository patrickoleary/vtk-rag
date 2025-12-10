"""Build Qdrant indexes for VTK code and documentation chunks.

Creates two collections:
- vtk_code: Code chunks from examples/tests with hybrid search
- vtk_docs: Class/method documentation chunks with hybrid search
"""

import json
import time
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    models,
)
from sentence_transformers import SentenceTransformer

from .collection_config import CODE_COLLECTION_CONFIG, DOC_COLLECTION_CONFIG, CollectionConfig


class Indexer:
    """Index VTK RAG chunks into Qdrant.

    Supports:
    - Dense vector search (semantic similarity)
    - Sparse vector search (BM25 keyword matching)
    - Hybrid search (dense + sparse with RRF fusion)
    - Metadata filtering (keyword, numeric)
    - Multi-vector indexing (content + pre-generated queries)
    """

    def __init__(
        self,
        base_path: Path | None = None,
        qdrant_url: str | None = None,
        dense_model: str | None = None,
        sparse_model: str | None = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            base_path: Project root directory. Defaults to vtk_rag parent.
            qdrant_url: Qdrant server URL. Defaults to config or localhost:6333.
            dense_model: Sentence transformer model for dense embeddings.
            sparse_model: FastEmbed model for sparse (BM25) embeddings.
        """
        # Load config for defaults
        from vtk_rag.config import get_config
        config = get_config()
        
        self.base_path = base_path or Path(__file__).parent.parent.parent
        self.data_dir = self.base_path / "data/processed"

        self.client = QdrantClient(url=qdrant_url or config.qdrant.url)
        self.dense_model = SentenceTransformer(dense_model or config.embedding.dense_model)
        self.sparse_model = SparseTextEmbedding(sparse_model or config.embedding.sparse_model)
        self.vector_size = self.dense_model.get_sentence_embedding_dimension()

    def create_collection(self, config: CollectionConfig, recreate: bool = True) -> None:
        """Create a Qdrant collection with configured indexes.

        Args:
            config: Collection configuration.
            recreate: If True, delete existing collection first.
        """
        collection_name = config.name

        # Delete existing if requested
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
                time.sleep(1)
            except Exception:
                pass

        # Create collection with dense and sparse vector config
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "content": VectorParams(size=self.vector_size, distance=Distance.COSINE),
                "queries": VectorParams(size=self.vector_size, distance=Distance.COSINE, multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        print(f"Created collection: {collection_name} (dense + sparse)")
        time.sleep(2)

        # Create payload indexes
        for field_config in config.indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_config.name,
                    field_schema=field_config.index_type,
                )
                print(f"  Created {field_config.index_type} index on {field_config.name}")
            except Exception as e:
                print(f"  Warning: Could not create index on {field_config.name}: {e}")

        print(f"Collection {collection_name} ready")

    def index_chunks(
        self,
        config: CollectionConfig,
        chunks: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Index chunks into a collection.

        Args:
            config: Collection configuration.
            chunks: List of chunk dictionaries.
            batch_size: Number of chunks per batch.

        Returns:
            Number of chunks indexed.
        """
        collection_name = config.name
        print(f"Indexing {len(chunks)} chunks into {collection_name}...")

        points = []

        for i, chunk in enumerate(chunks):
            # Generate dense content embedding
            content = chunk.get("content", "")
            content_vector = self.dense_model.encode(content).tolist()

            # Generate sparse BM25 embedding
            sparse_embeddings = list(self.sparse_model.embed([content]))[0]
            sparse_vector = SparseVector(
                indices=sparse_embeddings.indices.tolist(),
                values=sparse_embeddings.values.tolist(),
            )

            # Generate dense query embeddings (multi-vector)
            queries = chunk.get("queries", [])
            if queries:
                query_vectors = self.dense_model.encode(queries).tolist()
            else:
                # Fallback: use content as single query
                query_vectors = [content_vector]

            # Build payload (all fields except vectors)
            payload = {k: v for k, v in chunk.items() if k != "queries"}

            # Create point with dense and sparse vectors
            point = PointStruct(
                id=i,
                vector={
                    "content": content_vector,
                    "queries": query_vectors,
                    "bm25": sparse_vector,
                },
                payload=payload,
            )
            points.append(point)

            # Batch upload
            if len(points) >= batch_size:
                self.client.upsert(collection_name=collection_name, points=points)
                print(f"  Indexed {i + 1}/{len(chunks)} chunks")
                points = []

        # Upload remaining
        if points:
            self.client.upsert(collection_name=collection_name, points=points)

        print(f"Indexed {len(chunks)} chunks into {collection_name}")
        return len(chunks)

    def index_code(
        self,
        chunks_file: Path,
        recreate: bool = True,
        batch_size: int = 100,
    ) -> int:
        """Index code chunks from file.

        Args:
            chunks_file: Path to code-chunks.jsonl.
            recreate: If True, delete existing collection first.
            batch_size: Number of chunks per batch.

        Returns:
            Number of chunks indexed.
        """
        print(f"Indexing code from {chunks_file}")

        # Load chunks
        chunks = self._load_chunks(chunks_file)

        # Create collection
        self.create_collection(CODE_COLLECTION_CONFIG, recreate=recreate)

        # Index chunks
        return self.index_chunks(CODE_COLLECTION_CONFIG, chunks, batch_size=batch_size)

    def index_docs(
        self,
        chunks_file: Path,
        recreate: bool = True,
        batch_size: int = 100,
    ) -> int:
        """Index doc chunks from file.

        Args:
            chunks_file: Path to doc-chunks.jsonl.
            recreate: If True, delete existing collection first.
            batch_size: Number of chunks per batch.

        Returns:
            Number of chunks indexed.
        """
        print(f"Indexing docs from {chunks_file}")

        # Load chunks
        chunks = self._load_chunks(chunks_file)

        # Create collection
        self.create_collection(DOC_COLLECTION_CONFIG, recreate=recreate)

        # Index chunks
        return self.index_chunks(DOC_COLLECTION_CONFIG, chunks, batch_size=batch_size)

    def index_all(
        self,
        recreate: bool = True,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Index both code and doc chunks.

        Reads from data/processed/code-chunks.jsonl and doc-chunks.jsonl.

        Args:
            recreate: If True, delete existing collections first.
            batch_size: Number of chunks per batch.

        Returns:
            Dict mapping collection name to chunk count.
        """
        print("=" * 60)
        print("VTK RAG Indexing")
        print("=" * 60)
        print(f"Dense model: {self.dense_model.get_sentence_embedding_dimension()}-dim")
        print("Sparse model: BM25")

        results = {}

        code_file = self.data_dir / "code-chunks.jsonl"
        if code_file.exists():
            results["vtk_code"] = self.index_code(code_file, recreate, batch_size)
        else:
            print(f"Warning: Code chunks file not found: {code_file}")

        doc_file = self.data_dir / "doc-chunks.jsonl"
        if doc_file.exists():
            results["vtk_docs"] = self.index_docs(doc_file, recreate, batch_size)
        else:
            print(f"Warning: Doc chunks file not found: {doc_file}")

        # Summary
        print("\n" + "=" * 60)
        print("Indexing Complete")
        print("=" * 60)
        for collection, count in results.items():
            info = self.get_collection_info(collection)
            print(f"  {collection}: {count:,} chunks indexed ({info['status']})")
        print("\nQdrant dashboard: http://localhost:6333/dashboard")

        return results

    def _load_chunks(self, file_path: Path) -> list[dict[str, Any]]:
        """Load chunks from JSONL file.

        Args:
            file_path: Path to JSONL file.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the Qdrant collection.

        Returns:
            Dict with name, points_count, and status.
        """
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }
