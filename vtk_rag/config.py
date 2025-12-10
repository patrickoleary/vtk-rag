"""
Configuration management for VTK RAG.

Loads configuration from .env file and provides typed access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    url: str = "http://localhost:6333"
    code_collection: str = "vtk_code"
    docs_collection: str = "vtk_docs"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    dense_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"


@dataclass
class MCPConfig:
    """MCP server configuration for VTK API access."""
    vtk_api_docs_path: Path | None = None


@dataclass
class AppConfig:
    """Application configuration for VTK RAG."""
    
    # Qdrant configuration
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    
    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # MCP configuration
    mcp: MCPConfig = field(default_factory=MCPConfig)
    
    # Data paths
    data_dir: Path | None = None


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
    
    # Qdrant configuration
    qdrant_config = QdrantConfig(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        code_collection=os.getenv("QDRANT_CODE_COLLECTION", "vtk_code"),
        docs_collection=os.getenv("QDRANT_DOCS_COLLECTION", "vtk_docs"),
    )
    
    # Embedding configuration
    embedding_config = EmbeddingConfig(
        dense_model=os.getenv("EMBEDDING_DENSE_MODEL", "all-MiniLM-L6-v2"),
        sparse_model=os.getenv("EMBEDDING_SPARSE_MODEL", "Qdrant/bm25"),
    )
    
    # MCP configuration
    vtk_api_docs_env = os.getenv("VTK_API_DOCS_PATH")
    mcp_config = MCPConfig(
        vtk_api_docs_path=Path(vtk_api_docs_env) if vtk_api_docs_env else None,
    )
    
    # Data directory
    data_dir_env = os.getenv("VTK_RAG_DATA_DIR")
    data_dir = Path(data_dir_env) if data_dir_env else None
    
    return AppConfig(
        qdrant=qdrant_config,
        embedding=embedding_config,
        mcp=mcp_config,
        data_dir=data_dir,
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
