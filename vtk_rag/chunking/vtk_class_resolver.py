"""VTK class resolution via the vtkapi-mcp server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp import StdioServerParameters

from .persistent_mcp_client import PersistentMCPClient


class VTKClassResolver:
    """Resolve VTK class names to canonical modules via MCP."""

    def __init__(self) -> None:
        # Locate the authoritative VTK API docs (vtk-python-docs.jsonl).
        # Path: vtk_rag/chunking/vtk_class_resolver.py -> parents[2] = repo root
        repo_root = Path(__file__).resolve().parents[2]
        self.api_docs_path = repo_root / "data" / "raw" / "vtk-python-docs.jsonl"
        if not self.api_docs_path.exists():
            raise FileNotFoundError(
                f"VTK API docs not found at {self.api_docs_path}; cannot initialize MCP resolver"
            )
        # Configure the MCP server to run vtkapi-mcp with the docs file.
        self._server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "vtkapi_mcp", "--api-docs", str(self.api_docs_path)],
        )
        # Start a persistent MCP client (launches server once, reuses session).
        self._client = PersistentMCPClient(self._server)
        # Cache class→module mappings to avoid redundant MCP queries.
        self._cache: dict[str, str] = {}

    def _query_classes(self, class_names: set[str]) -> dict[str, str]:
        """Issue MCP queries for the provided class names via the persistent session."""
        modules: dict[str, str] = {}
        for class_name in sorted(class_names):
            result = self._client.call_tool("vtk_get_class_info", {"class_name": class_name})
            # Parse the JSON response to extract the canonical module name.
            payload = json.loads(result.content[0].text)
            module = payload.get("module")
            if module:
                modules[class_name] = module
        return modules

    def resolve(self, class_names: set[str]) -> dict[str, str]:
        """Return class→module map, reusing cached entries and querying MCP for cache misses."""
        # Identify which class names are not yet in the cache.
        pending = {name for name in class_names if name not in self._cache}
        if pending:
            # Query the MCP server for the missing classes and update the cache.
            resolved = self._query_classes(pending)
            self._cache.update(resolved)
        # Return only the successfully resolved classes from the cache.
        return {name: self._cache[name] for name in class_names if name in self._cache}

    def get_class_info(self, class_name: str) -> dict | None:
        """Get full class info from MCP."""
        try:
            result = self._client.call_tool("vtk_get_class_info", {"class_name": class_name})
            if result and result.content:
                return json.loads(result.content[0].text)
        except Exception:
            pass
        return None

    def get_method_info(self, class_name: str, method_name: str) -> dict | None:
        """Get method info from MCP."""
        try:
            result = self._client.call_tool(
                "vtk_get_method_info",
                {"class_name": class_name, "method_name": method_name}
            )
            if result and result.content:
                return json.loads(result.content[0].text)
        except Exception:
            pass
        return None

    def get_class_role(self, class_name: str) -> str | None:
        """Get class role (e.g., 'source', 'filter', 'mapper') from MCP."""
        try:
            result = self._client.call_tool("vtk_get_class_role", {"class_name": class_name})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("role")
        except Exception:
            pass
        return None

    def get_class_visibility(self, class_name: str) -> str | None:
        """Get class visibility string (e.g., 'very_likely', 'likely', 'maybe') from MCP."""
        try:
            result = self._client.call_tool("vtk_get_class_visibility", {"class_name": class_name})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                if payload.get("found"):
                    return payload.get("visibility")
        except Exception:
            pass
        return None

    def get_class_action_phrase(self, class_name: str) -> str | None:
        """Get action phrase for a class (e.g., 'polygonal sphere creation') from MCP."""
        try:
            result = self._client.call_tool("vtk_get_class_action_phrase", {"class_name": class_name})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("action_phrase")
        except Exception:
            pass
        return None


# Global singleton: one persistent MCP session shared across all metadata extractions.
VTK_CLASS_RESOLVER = VTKClassResolver()
