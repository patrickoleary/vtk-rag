"""RAG client providing Qdrant and embedding access.

Used by:
    - chunking/chunker.py (Chunker)
    - indexing/indexer.py (Indexer)
    - retrieval/retriever.py (Retriever)

Code Map:
    RAGClient
        __init__()                   # initialize Qdrant + embedding models
        qdrant_url                   # property: Qdrant server URL
        code_collection              # property: code collection name
        docs_collection              # property: docs collection name
        top_k                        # property: default result limit
        use_hybrid                   # property: hybrid search flag
        min_visibility_score         # property: visibility threshold
        raw_dir                      # property: raw data directory
        chunk_dir                    # property: chunk output directory
        examples_file                # property: examples filename
        tests_file                   # property: tests filename
        docs_file                    # property: docs filename
        code_chunks                  # property: code chunks filename
        doc_chunks                   # property: doc chunks filename
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from vtk_rag.config import AppConfig, RAGConfig


class RAGClient:
    """Client providing Qdrant and embedding access for RAG operations.

    Instantiated from AppConfig and passed to Retriever, Indexer, Chunker.

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

    def __init__(self, config: AppConfig) -> None:
        """Initialize the RAG client.

        Args:
            config: Application configuration.
        """
        self.config = config.rag_client

        # Qdrant client
        self.qdrant_client = QdrantClient(url=self.config.qdrant_url)

        # Embedding models
        self.dense_model = SentenceTransformer(self.config.dense_model)
        self.sparse_model = SparseTextEmbedding(self.config.sparse_model)

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
