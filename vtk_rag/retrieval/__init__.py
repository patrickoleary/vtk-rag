"""VTK RAG Retrieval Module.

Core retrieval primitives for searching VTK code and documentation.
"""

from .models import SearchResult
from .retriever import Retriever

__all__ = [
    "Retriever",
    "SearchResult",
]
