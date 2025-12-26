"""VTK API client using direct VTKAPIIndex access (no MCP subprocess).

Used by:
    - chunking/chunker.py (Chunker)
    - chunking/code/chunker.py (CodeChunker)
    - chunking/code/semantic_chunk.py (SemanticChunk)
    - chunking/code/lifecycle/analyzer.py (LifecycleAnalyzer)
    - chunking/code/lifecycle/builder.py (build_lifecycles)
    - chunking/code/lifecycle/visitor.py (LifecycleVisitor)
    - chunking/doc/chunker.py (DocChunker)
    - chunking/query/semantic_query.py (SemanticQuery)

Code Map:
    VTKClient
        is_vtk_class()               # check if class name is VTK
        get_class_module()           # module path for class
        get_class_modules()          # module paths for multiple classes
        get_class_info()             # complete class metadata
        get_class_role()             # functional role (input, filter, etc.)
        get_class_visibility()       # visibility score (0.0-1.0)
        get_class_action_phrase()    # action phrase for class
        get_class_synopsis()         # brief synopsis
        get_class_doc()              # class docstring
        get_class_methods()          # list of methods with signatures
        get_class_input_datatype()   # input data type
        get_class_output_datatype()  # output data type
        get_class_semantic_methods() # non-boilerplate methods
        get_method_info()            # method metadata
        get_method_doc()             # method docstring
        get_method_signature()       # method signature
        get_module_classes()         # classes in a module
        search_classes()             # search by name/keyword
    get_vtk_client()                 # singleton factory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtkapi_mcp import VTKAPIIndex

from vtk_rag.config import AppConfig


class VTKClient:
    """Direct VTK API client using VTKAPIIndex.

    Provides methods to query VTK class metadata, method signatures, and
    documentation. Uses direct in-memory access instead of MCP subprocess.

    Attributes:
        api_docs_path: Resolved path to VTK API docs JSONL file.
    """

    api_docs_path: Path
    _index: VTKAPIIndex

    def __init__(self, config: AppConfig) -> None:
        """Initialize the VTK API client.

        Args:
            config: Application configuration containing MCP settings.

        Raises:
            ValueError: If MCPConfig.vtk_api_docs_path is not set.
            FileNotFoundError: If VTK API docs file doesn't exist.
        """
        mcp_config = config.mcp

        if mcp_config.vtk_api_docs_path is None:
            raise ValueError("MCPConfig.vtk_api_docs_path must be set")

        api_docs_path = mcp_config.vtk_api_docs_path
        if not api_docs_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            api_docs_path = repo_root / api_docs_path

        self.api_docs_path = api_docs_path
        if not self.api_docs_path.exists():
            raise FileNotFoundError(
                f"VTK API docs not found at {self.api_docs_path}"
            )

        self._index = VTKAPIIndex(self.api_docs_path)

    def is_vtk_class(self, class_name: str) -> bool:
        """Check if a class name is a known VTK class.

        Args:
            class_name: Class name to check.

        Returns:
            True if class_name is a known VTK class.
        """
        return class_name in self._index.classes

    def get_class_module(self, class_name: str) -> str | None:
        """Get the module path for a VTK class.

        Args:
            class_name: VTK class name (e.g., 'vtkSphereSource').

        Returns:
            Module path (e.g., 'vtkmodules.vtkFiltersSources') or None.
        """
        return self._index.get_class_module(class_name)

    def get_class_modules(self, class_names: set[str]) -> dict[str, str]:
        """Get module paths for multiple VTK classes.

        Args:
            class_names: Set of VTK class names.

        Returns:
            Dict mapping class names to their module paths.
        """
        result = {}
        for name in class_names:
            module = self._index.get_class_module(name)
            if module:
                result[name] = module
        return result

    def get_class_info(self, class_name: str) -> dict[str, Any] | None:
        """Get complete information about a VTK class.

        Args:
            class_name: VTK class name.

        Returns:
            Dict with class metadata, or None if not found.
        """
        return self._index.get_class_info(class_name)

    def get_class_role(self, class_name: str) -> str | None:
        """Get the functional role of a VTK class.

        Args:
            class_name: VTK class name.

        Returns:
            Role string (e.g., 'input', 'filter', 'properties'), or None.
        """
        return self._index.get_class_role(class_name)

    def get_class_visibility(self, class_name: str) -> float | None:
        """Get class visibility score.

        Args:
            class_name: VTK class name.

        Returns:
            Visibility score (0.0-1.0), or None if not found.
        """
        result = self._index.get_class_visibility(class_name)
        if result is not None:
            return float(result) if result else None
        return None

    def get_class_action_phrase(self, class_name: str) -> str | None:
        """Get action phrase for a class.

        Args:
            class_name: VTK class name.

        Returns:
            Action phrase (e.g., 'polygonal sphere creation'), or None.
        """
        return self._index.get_class_action_phrase(class_name)

    def get_class_synopsis(self, class_name: str) -> str | None:
        """Get brief synopsis of what a VTK class does.

        Args:
            class_name: VTK class name.

        Returns:
            Synopsis string, or None.
        """
        return self._index.get_class_synopsis(class_name)

    def get_class_doc(self, class_name: str) -> str | None:
        """Get the class documentation string.

        Args:
            class_name: VTK class name.

        Returns:
            Class docstring, or None.
        """
        return self._index.get_class_doc(class_name)

    def get_class_methods(self, class_name: str) -> list[dict[str, Any]]:
        """Get structured list of methods with signatures.

        Args:
            class_name: VTK class name.

        Returns:
            List of method dicts with 'method_name', 'signature', 'doc' keys.
        """
        return self._index.get_class_methods(class_name) or []

    def get_class_input_datatype(self, class_name: str) -> str | None:
        """Get the input data type for a VTK class.

        Args:
            class_name: VTK class name.

        Returns:
            Input data type (e.g., 'vtkDataSet'), or None.
        """
        return self._index.get_class_input_datatype(class_name)

    def get_class_output_datatype(self, class_name: str) -> str | None:
        """Get the output data type for a VTK class.

        Args:
            class_name: VTK class name.

        Returns:
            Output data type (e.g., 'vtkPolyData'), or None.
        """
        return self._index.get_class_output_datatype(class_name)

    def get_class_semantic_methods(self, class_name: str) -> list[str]:
        """Get non-boilerplate callable methods for a VTK class.

        Args:
            class_name: VTK class name.

        Returns:
            Sorted list of semantic method names.
        """
        return self._index.get_class_semantic_methods(class_name) or []

    def get_method_info(self, class_name: str, method_name: str) -> dict[str, Any] | None:
        """Get information about a specific method.

        Args:
            class_name: VTK class name.
            method_name: Method name.

        Returns:
            Dict with method metadata, or None.
        """
        return self._index.get_method_info(class_name, method_name)

    def get_method_doc(self, class_name: str, method_name: str) -> str | None:
        """Get the docstring for a specific method.

        Args:
            class_name: VTK class name.
            method_name: Method name.

        Returns:
            Method docstring, or None.
        """
        return self._index.get_method_doc(class_name, method_name)

    def get_method_signature(self, class_name: str, method_name: str) -> str | None:
        """Get the canonical signature for a method.

        Args:
            class_name: VTK class name.
            method_name: Method name.

        Returns:
            Signature string (e.g., 'GetOutput(self) -> vtkPolyData'), or None.
        """
        return self._index.get_method_signature(class_name, method_name)

    def get_module_classes(self, module: str) -> list[str]:
        """List all VTK classes in a specific module.

        Args:
            module: Module name (e.g., 'vtkmodules.vtkRenderingCore').

        Returns:
            List of class names in the module.
        """
        return self._index.get_module_classes(module)

    def search_classes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Search for VTK classes by name or keyword.

        Args:
            query: Search term.
            limit: Maximum number of results.

        Returns:
            List of dicts with 'class_name', 'module', 'description'.
        """
        return self._index.search_classes(query, limit)


# Lazy-loaded singleton
_vtk_client: VTKClient | None = None


def get_vtk_client(config: AppConfig | None = None) -> VTKClient:
    """Get or create the global VTK client singleton.

    Args:
        config: Application configuration. If None, uses get_config().

    Returns:
        VTKClient instance.

    Raises:
        FileNotFoundError: If VTK API docs file doesn't exist.
        ValueError: If MCPConfig.vtk_api_docs_path is not set.
    """
    global _vtk_client
    if _vtk_client is None:
        if config is None:
            from vtk_rag.config import get_config
            config = get_config()
        _vtk_client = VTKClient(config)
    return _vtk_client
