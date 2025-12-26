"""Build Qdrant indexes for VTK code and documentation chunks.

Creates two collections:
- vtk_code: Code chunks from examples/tests with hybrid search
- vtk_docs: Class/method documentation chunks with hybrid search

Code Map:
    Indexer
        index()                      # public API - index all collections
            ├── _index_collection()  # index single collection
            │       ├── _load_chunks()        # load JSONL file
            │       ├── _create_collection()  # create Qdrant collection
            │       └── _index_chunks()       # batch upload points
            └── _get_collection_info()        # collection status
"""

import json
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    models,
)

from vtk_rag.rag import RAGClient

from .models import CODE_CONFIG, DOC_CONFIG, CollectionConfig


class Indexer:
    """Index VTK RAG chunks into Qdrant.

    Attributes:
        rag_client: RAG client providing Qdrant and embedding access.
        qdrant_client: Qdrant client for vector database operations.
        base_path: Project root directory.
    """

    rag_client: RAGClient
    qdrant_client: QdrantClient
    base_path: Path

    def __init__(
        self,
        rag_client: RAGClient,
        base_path: Path | None = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            rag_client: RAG client providing Qdrant and embedding access.
            base_path: Project root directory. Defaults to vtk_rag parent.
        """
        self.rag_client = rag_client
        self.qdrant_client = rag_client.qdrant_client
        self.base_path = base_path or Path(__file__).parent.parent.parent

    def index(
        self,
        recreate: bool = True,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Index both code and doc chunks.

        Reads chunk files from rag_client.chunk_dir.

        Args:
            recreate: If True, delete existing collections first.
            batch_size: Number of chunks per batch.

        Returns:
            Dict mapping collection name to chunk count.
        """
        print("=" * 60)
        print("VTK RAG Indexing")
        print("=" * 60)
        print(f"Dense model: {self.rag_client.dense_model.get_sentence_embedding_dimension()}-dim")
        print("Sparse model: BM25")

        results = {}

        code_file = self.base_path / self.rag_client.chunk_dir / self.rag_client.code_chunks
        if code_file.exists():
            results["vtk_code"] = self._index_collection(CODE_CONFIG, code_file, recreate, batch_size)
        else:
            print(f"Warning: Code chunks file not found: {code_file}")

        doc_file = self.base_path / self.rag_client.chunk_dir / self.rag_client.doc_chunks
        if doc_file.exists():
            results["vtk_docs"] = self._index_collection(DOC_CONFIG, doc_file, recreate, batch_size)
        else:
            print(f"Warning: Doc chunks file not found: {doc_file}")

        # Summary
        print("\n" + "=" * 60)
        print("Indexing Complete")
        print("=" * 60)
        for collection, count in results.items():
            info = self._get_collection_info(collection)
            print(f"  {collection}: {count:,} chunks indexed ({info['status']})")
        print("\nQdrant dashboard: http://localhost:6333/dashboard")
        return results

    def _index_collection(
        self,
        config: CollectionConfig,
        chunks_file: Path,
        recreate: bool = True,
        batch_size: int = 100,
    ) -> int:
        """Index chunks from file into a collection.

        Args:
            config: Collection configuration (CODE_CONFIG or DOC_CONFIG).
            chunks_file: Path to chunks JSONL file.
            recreate: If True, delete existing collection first.
            batch_size: Number of chunks per batch.

        Returns:
            Number of chunks indexed.
        """
        print(f"Indexing {config.name} from {chunks_file}")
        chunks = self._load_chunks(chunks_file)
        self._create_collection(config, recreate=recreate)
        return self._index_chunks(config, chunks, batch_size=batch_size)

    def _load_chunks(self, file_path: Path) -> list[dict[str, Any]]:
        """Load chunks from JSONL file.

        Args:
            file_path: Path to JSONL file.

        Returns:
            List of chunk dictionaries.
        """
        with open(file_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks

    def _create_collection(self, config: CollectionConfig, recreate: bool = True) -> None:
        """Create a Qdrant collection with configured indexes.

        Args:
            config: Collection configuration.
            recreate: If True, delete existing collection first.
        """
        collection_name = config.name

        # Delete existing if requested
        if recreate:
            try:
                self.qdrant_client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
                time.sleep(1)
            except Exception:
                pass

        # Create collection with dense and sparse vector config
        vector_size = self.rag_client.dense_model.get_sentence_embedding_dimension()
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "content": VectorParams(size=vector_size, distance=Distance.COSINE),
                "queries": VectorParams(size=vector_size, distance=Distance.COSINE, multivector_config=models.MultiVectorConfig(
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
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_config.name,
                    field_schema=field_config.index_type,
                )
                print(f"  Created {field_config.index_type} index on {field_config.name}")
            except Exception as e:
                print(f"  Warning: Could not create index on {field_config.name}: {e}")

        print(f"Collection {collection_name} ready")

    def _index_chunks(
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
            content_vector = self.rag_client.dense_model.encode(content).tolist()

            # Generate sparse BM25 embedding
            sparse_embeddings = list(self.rag_client.sparse_model.embed([content]))[0]
            sparse_vector = SparseVector(
                indices=sparse_embeddings.indices.tolist(),
                values=sparse_embeddings.values.tolist(),
            )

            # Generate dense query embeddings (multi-vector)
            queries = chunk.get("queries", [])
            if queries:
                query_vectors = self.rag_client.dense_model.encode(queries).tolist()
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
                self.qdrant_client.upsert(collection_name=collection_name, points=points)
                print(f"  Indexed {i + 1}/{len(chunks)} chunks")
                points = []

        # Upload remaining
        if points:
            self.qdrant_client.upsert(collection_name=collection_name, points=points)

        print(f"Indexed {len(chunks)} chunks into {collection_name}")
        return len(chunks)

    def _get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the Qdrant collection.

        Returns:
            Dict with name, points_count, and status.
        """
        info = self.qdrant_client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }
