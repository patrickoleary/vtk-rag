"""
Configuration management for VTK RAG.

Loads configuration from .env file and provides typed access.

Used by:
    - cli.py, build.py (get_config)
    - mcp/client.py (AppConfig, get_config)
    - rag/client.py (RAGConfig)

Code Map:
    MCPConfig                        # MCP server settings
    RAGConfig                        # Qdrant, embedding, data path settings
    AppConfig                        # top-level config container
    load_config()                    # load from .env file
    get_config()                     # singleton accessor
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class MCPConfig:
    """MCP server configuration for VTK API access."""
    vtk_api_docs_path: Path | None = None
    max_tool_iterations: int = 10

@dataclass
class RAGConfig:
    """RAG retrieval configuration.

    Consolidates Qdrant, embedding, and data path settings.
    """
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    code_collection: str = "vtk_code"
    docs_collection: str = "vtk_docs"

    # Embedding models
    dense_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"

    # Search settings
    top_k: int = 5
    use_hybrid: bool = True
    min_visibility_score: float = 0.0

    # Data paths (relative to project root or absolute)
    raw_dir: str = "data/raw"
    chunk_dir: str = "data/processed"
    examples_file: str = "vtk-python-examples.jsonl"
    tests_file: str = "vtk-python-tests.jsonl"
    docs_file: str = "vtk-python-docs.jsonl"
    code_chunks: str = "code-chunks.jsonl"
    doc_chunks: str = "doc-chunks.jsonl"

@dataclass
class AppConfig:
    """Application configuration for VTK RAG."""

    # RAG client configuration (Qdrant + embeddings)
    rag_client: RAGConfig = field(default_factory=RAGConfig)

    # MCP configuration
    mcp: MCPConfig = field(default_factory=MCPConfig)



def load_config(env_path: Path | None = None) -> AppConfig:
    """
    Load configuration from .env file.

    Args:
        env_path: Optional path to .env file. If None, searches current directory
                  and parent directories.

    Returns:
        AppConfig with values from environment
    """
    # Load .env file
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    # MCP configuration
    mcp_config = MCPConfig(
        vtk_api_docs_path=Path(os.getenv("VTK_API_DOCS_PATH", "data/vtk-python-docs.jsonl")),
        max_tool_iterations=int(os.getenv("MCP_MAX_TOOL_ITERATIONS", "10")),
    )

    # RAG configuration
    rag_config = RAGConfig(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        code_collection=os.getenv("QDRANT_CODE_COLLECTION", "vtk_code"),
        docs_collection=os.getenv("QDRANT_DOCS_COLLECTION", "vtk_docs"),
        dense_model=os.getenv("EMBEDDING_DENSE_MODEL", "all-MiniLM-L6-v2"),
        sparse_model=os.getenv("EMBEDDING_SPARSE_MODEL", "Qdrant/bm25"),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        use_hybrid=os.getenv("RAG_USE_HYBRID", "true").lower() == "true",
        min_visibility_score=float(os.getenv("RAG_MIN_VISIBILITY", "0.0")),
        raw_dir=os.getenv("RAG_RAW_DIR", "data/raw"),
        chunk_dir=os.getenv("RAG_CHUNK_DIR", "data/processed"),
        examples_file=os.getenv("RAG_EXAMPLES_FILE", "vtk-python-examples.jsonl"),
        tests_file=os.getenv("RAG_TESTS_FILE", "vtk-python-tests.jsonl"),
        docs_file=os.getenv("RAG_DOCS_FILE", "vtk-python-docs.jsonl"),
        code_chunks=os.getenv("RAG_CODE_CHUNKS", "code-chunks.jsonl"),
        doc_chunks=os.getenv("RAG_DOC_CHUNKS", "doc-chunks.jsonl"),
    )

    return AppConfig(
        rag_client=rag_config,
        mcp=mcp_config,
    )


# Global config instance (lazy-loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Loads from .env on first call, then returns cached instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
