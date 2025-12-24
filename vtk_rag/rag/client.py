"""RAG client providing Qdrant and embedding access."""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from vtk_rag.config import RAGConfig


class RAGClient:
    """Client providing Qdrant and embedding access for RAG operations.

    Instantiated from RAGConfig and passed to Retriever, Indexer, Chunker.

    Attributes:
        config: The RAGConfig used to initialize this client.
        qdrant_client: Qdrant client for vector database operations.
        dense_model: SentenceTransformer for dense embeddings.
        sparse_model: FastEmbed model for sparse (BM25) embeddings.

    See Also:
        examples/rag_client_usage.py for usage examples.
    """

    config: RAGConfig
    qdrant_client: QdrantClient
    dense_model: SentenceTransformer
    sparse_model: SparseTextEmbedding

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the RAG client.

        Args:
            config: RAG configuration (from AppConfig.rag_client).
        """
        self.config = config

        # Qdrant client
        self.qdrant_client = QdrantClient(url=config.qdrant_url)

        # Embedding models
        self.dense_model = SentenceTransformer(config.dense_model)
        self.sparse_model = SparseTextEmbedding(config.sparse_model)

    @property
    def qdrant_url(self) -> str:
        return self.config.qdrant_url

    @property
    def code_collection(self) -> str:
        return self.config.code_collection

    @property
    def docs_collection(self) -> str:
        return self.config.docs_collection

    @property
    def top_k(self) -> int:
        return self.config.top_k

    @property
    def use_hybrid(self) -> bool:
        return self.config.use_hybrid

    @property
    def min_visibility_score(self) -> float:
        return self.config.min_visibility_score

    @property
    def raw_dir(self) -> str:
        return self.config.raw_dir

    @property
    def chunk_dir(self) -> str:
        return self.config.chunk_dir

    @property
    def examples_file(self) -> str:
        return self.config.examples_file

    @property
    def tests_file(self) -> str:
        return self.config.tests_file

    @property
    def docs_file(self) -> str:
        return self.config.docs_file

    @property
    def code_chunks(self) -> str:
        return self.config.code_chunks

    @property
    def doc_chunks(self) -> str:
        return self.config.doc_chunks
