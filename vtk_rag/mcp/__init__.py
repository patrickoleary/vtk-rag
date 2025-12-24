"""VTK RAG MCP Module.

VTK API client utilities using direct VTKAPIIndex access.
"""

from .client import VTKClient, get_vtk_client

__all__ = [
    "VTKClient",
    "get_vtk_client",
]
