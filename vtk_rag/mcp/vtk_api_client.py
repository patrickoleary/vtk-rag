"""VTK class resolution via the vtkapi-mcp server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp import StdioServerParameters

from .persistent_mcp_client import PersistentMCPClient


class VTKAPIClient:
    """Resolve VTK class names to canonical modules via MCP."""

    def __init__(self, api_docs_path: Path | None = None) -> None:
        """Initialize the VTK API client.
        
        Args:
            api_docs_path: Path to VTK API docs JSONL file. If None, uses config
                          or falls back to default location.
        """
        # Resolve API docs path from: explicit arg > config > default
        if api_docs_path is None:
            from vtk_rag.config import get_config
            config = get_config()
            api_docs_path = config.mcp.vtk_api_docs_path
        
        if api_docs_path is None:
            # Default: repo_root/data/raw/vtk-python-docs.jsonl
            repo_root = Path(__file__).resolve().parents[2]
            api_docs_path = repo_root / "data" / "raw" / "vtk-python-docs.jsonl"
        
        # Resolve relative paths against repo root
        if not api_docs_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            api_docs_path = repo_root / api_docs_path
        
        self.api_docs_path = api_docs_path
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

    def search_classes(self, query: str, limit: int = 10) -> list[str]:
        """Search for VTK classes by name or keyword.
        
        Args:
            query: Search term (e.g., 'reader', 'mapper', 'actor')
            limit: Maximum number of results
            
        Returns:
            List of matching class names
        """
        try:
            result = self._client.call_tool(
                "vtk_search_classes",
                {"query": query, "limit": limit}
            )
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("classes", [])
        except Exception:
            pass
        return []

    def get_module_classes(self, module: str) -> list[str]:
        """List all VTK classes in a specific module.
        
        Args:
            module: Module name (e.g., 'vtkmodules.vtkRenderingCore')
            
        Returns:
            List of class names in the module
        """
        try:
            result = self._client.call_tool("vtk_get_module_classes", {"module": module})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("classes", [])
        except Exception:
            pass
        return []

    def validate_import(self, import_statement: str) -> dict | None:
        """Validate if a VTK import statement is correct and suggest corrections.
        
        Args:
            import_statement: Python import statement to validate
            
        Returns:
            Dict with validation result and suggestions, or None on error
        """
        try:
            result = self._client.call_tool(
                "vtk_validate_import",
                {"import_statement": import_statement}
            )
            if result and result.content:
                return json.loads(result.content[0].text)
        except Exception:
            pass
        return None

    def get_method_doc(self, class_name: str, method_name: str) -> str | None:
        """Get the docstring for a specific method of a VTK class.
        
        Args:
            class_name: VTK class name
            method_name: Method name
            
        Returns:
            Method docstring or None
        """
        try:
            result = self._client.call_tool(
                "vtk_get_method_doc",
                {"class_name": class_name, "method_name": method_name}
            )
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("doc")
        except Exception:
            pass
        return None

    def get_class_doc(self, class_name: str) -> str | None:
        """Get the class documentation string for a VTK class.
        
        Args:
            class_name: VTK class name
            
        Returns:
            Class docstring or None
        """
        try:
            result = self._client.call_tool("vtk_get_class_doc", {"class_name": class_name})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("doc")
        except Exception:
            pass
        return None

    def get_class_synopsis(self, class_name: str) -> str | None:
        """Get a brief synopsis/summary of what a VTK class does.
        
        Args:
            class_name: VTK class name
            
        Returns:
            Synopsis string or None
        """
        try:
            result = self._client.call_tool("vtk_get_class_synopsis", {"class_name": class_name})
            if result and result.content:
                payload = json.loads(result.content[0].text)
                return payload.get("synopsis")
        except Exception:
            pass
        return None


# Global singleton: one persistent MCP session shared across all metadata extractions.
VTK_API_CLIENT = VTKAPIClient()
