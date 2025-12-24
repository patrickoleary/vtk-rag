"""Query generation for VTK chunks.

Provides SemanticQuery for generating natural language "how do you" queries
from VTK class action phrases and method calls.
"""

from .semantic_query import SemanticQuery

__all__ = ["SemanticQuery"]
