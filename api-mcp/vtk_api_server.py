#!/usr/bin/env python3
"""
VTK API MCP Server

Provides direct access to VTK API documentation through MCP tools.
Replaces the need for API docs in RAG retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from mcp.server import Server
from mcp.types import Tool, TextContent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VTKAPIIndex:
    """Fast in-memory index of VTK API documentation"""
    
    def __init__(self, api_docs_path: Path):
        """
        Initialize the API index
        
        Args:
            api_docs_path: Path to vtk-python-docs.jsonl (raw, not chunked)
        """
        self.api_docs_path = api_docs_path
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.modules: Dict[str, List[str]] = {}  # module -> [class names]
        self._load_api_docs()
    
    def _load_api_docs(self):
        """Load all API documentation from raw vtk-python-docs.jsonl"""
        logger.info(f"Loading VTK API docs from {self.api_docs_path}")
        
        if not self.api_docs_path.exists():
            logger.error(f"API docs not found at {self.api_docs_path}")
            return
        
        with open(self.api_docs_path) as f:
            for line in f:
                doc = json.loads(line)
                
                # Raw format: each line is a complete class documentation
                class_name = doc.get('class_name')
                
                if not class_name:
                    continue
                
                # Get content (full documentation with all methods)
                content = doc.get('content', '')
                module = doc.get('module_name', '')  # Raw format uses 'module_name' not 'module'
                
                # Store class info
                self.classes[class_name] = {
                    'class_name': class_name,
                    'module': module,
                    'content': content,
                    'metadata': doc
                }
                
                # Index by module
                if module:
                    if module not in self.modules:
                        self.modules[module] = []
                    self.modules[module].append(class_name)
        
        logger.info(f"Loaded {len(self.classes)} VTK classes from {len(self.modules)} modules")
    
    def _extract_module(self, content: str) -> Optional[str]:
        """Extract module path from class content"""
        # Look for "**Module:** `vtkmodules.XXX`"
        if '**Module:**' in content:
            lines = content.split('\n')
            for line in lines:
                if '**Module:**' in line:
                    # Extract module from markdown code
                    if '`' in line:
                        parts = line.split('`')
                        if len(parts) >= 2:
                            return parts[1].strip()
        return None
    
    def get_class_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get complete information about a VTK class"""
        return self.classes.get(class_name)
    
    def search_classes(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for classes by name or keyword
        
        Returns list of {class_name, module, description}
        """
        query_lower = query.lower()
        results = []
        
        for class_name, info in self.classes.items():
            # Match by class name
            if query_lower in class_name.lower():
                content = info['content']
                # Extract first line of description
                description = self._extract_description(content)
                
                results.append({
                    'class_name': class_name,
                    'module': info['module'] or 'Unknown',
                    'description': description
                })
        
        return results[:limit]
    
    def _extract_description(self, content: str) -> str:
        """Extract brief description from class content"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Find first non-header, non-module line
            if line.strip() and not line.startswith('#') and '**Module:**' not in line:
                # Take first sentence or first 100 chars
                desc = line.strip()
                if '.' in desc:
                    desc = desc.split('.')[0] + '.'
                return desc[:150]
    
    def get_module_classes(self, module: str) -> List[str]:
        """Get all classes in a module"""
        return self.modules.get(module, [])
    
    def _extract_used_classes(self, code: str) -> List[str]:
        """Extract all VTK class names that are actually used in the code"""
        import re
        used_classes = set()
        
        # Pattern 1: Class instantiation - vtkClassName()
        pattern1 = r'\b(vtk[A-Z]\w+)\s*\('
        for match in re.finditer(pattern1, code):
            used_classes.add(match.group(1))
        
        # Pattern 2: Class usage after import - ClassName() where ClassName starts with vtk
        # This catches usage in code body
        lines = code.split('\n')
        for line in lines:
            # Skip import lines
            if 'import' in line:
                continue
            # Find vtk class usage
            for match in re.finditer(r'\b(vtk[A-Z]\w+)', line):
                class_name = match.group(1)
                # Make sure it's actually a class (in our database)
                if class_name in self.classes:
                    used_classes.add(class_name)
        
        return list(used_classes)
    
    def validate_import(self, import_statement: str, code_context: str = None) -> Dict[str, Any]:
        """
        Validate if an import statement is correct
        
        Accepts three styles:
        - import vtk (monolithic)
        - import vtkmodules.all as vtk (modular all-in-one)
        - from vtkmodules.XXX import ClassName (modular selective)
        
        Returns: {valid: bool, message: str, suggested: str}
        """
        import_statement = import_statement.strip()
        
        # Style 1: "import vtk" - Always valid (monolithic import)
        if import_statement == 'import vtk':
            return {
                'valid': True,
                'message': 'Monolithic VTK import (valid)',
                'suggested': None
            }
        
        # Style 2: "import vtkmodules.all as vtk" - Valid (modular all-in-one)
        if import_statement == 'import vtkmodules.all as vtk':
            return {
                'valid': True,
                'message': 'Modular all-in-one VTK import (valid)',
                'suggested': None
            }
        
        # Style 3: "import vtkmodules.XXX" - Direct module import (ONLY for backend/rendering modules)
        import_clean = import_statement.split('#')[0].strip()  # Remove inline comments
        if import_clean.startswith('import vtkmodules.'):
            # Extract module name
            module_name = import_clean.replace('import ', '').strip()
            
            # ONLY allow direct import for specific backend modules that MUST be loaded this way
            allowed_direct_imports = {
                'vtkmodules.vtkRenderingOpenGL2',  # Required for offscreen rendering
                'vtkmodules.vtkInteractionStyle',   # Interaction backend
                'vtkmodules.vtkRenderingFreeType',  # Font rendering backend
                'vtkmodules.vtkRenderingVolumeOpenGL2',  # Volume rendering backend
            }
            
            if module_name in allowed_direct_imports:
                return {
                    'valid': True,
                    'message': f'Backend module import (valid - required for initialization)',
                    'suggested': None
                }
            else:
                # This module should be imported using "from vtkmodules.XXX import ClassName" style
                return {
                    'valid': False,
                    'message': (
                        f"INVALID: Direct module import not allowed.\n"
                        f"  Use 'from {module_name} import ClassName' instead.\n\n"
                        f"  SMALLEST CHANGE: Replace with proper from-import style"
                    ),
                    'suggested': f"from {module_name} import <ClassName>"
                }
        
        # Style 4: "from vtkmodules.XXX import ClassName" - Validate module path
        if 'from' in import_statement and 'import' in import_statement:
            parts = import_statement.split('import')
            if len(parts) == 2:
                class_part = parts[1].strip()
                # Handle multiple imports or parentheses
                class_names = []
                if '(' in class_part:
                    class_part = class_part.replace('(', '').replace(')', '')
                for name in class_part.split(','):
                    class_names.append(name.strip())
                
                # Check ALL classes on the import line
                if class_names:
                    module_part_from = parts[0].replace('from', '').strip()
                    
                    # Collect validation results for all imports
                    modules_to_delete = []
                    modules_with_usage = []
                    
                    for class_name in class_names:
                        full_name = f"{module_part_from}.{class_name}"
                        possible_module = f"vtkmodules.{class_name}"
                        
                        # Check if this is a module (not a class)
                        if full_name in self.modules or possible_module in self.modules:
                            # It's a module - check usage
                            if code_context:
                                used_classes = self._extract_used_classes(code_context)
                                module_classes = self.get_module_classes(possible_module)
                                classes_from_module = [c for c in used_classes if c in module_classes]
                                
                                if classes_from_module:
                                    modules_with_usage.append((class_name, possible_module, classes_from_module))
                                else:
                                    modules_to_delete.append(class_name)
                    
                    # Generate combined error message if any modules were found
                    if modules_to_delete or modules_with_usage:
                        # Build combined message for all module imports on this line
                        if modules_to_delete and not modules_with_usage:
                            # ALL modules are unused - just delete the line
                            module_list = ', '.join(modules_to_delete)
                            return {
                                'valid': False,
                                'message': (
                                    f"INVALID: Cannot import modules this way.\n"
                                    f"  These are MODULES, not classes: {module_list}\n\n"
                                    f"  Your code does NOT use any classes from these modules.\n\n"
                                    f"  SMALLEST CHANGE: DELETE this entire line (unused code):\n"
                                    f"    {import_statement.strip()}\n"
                                    f"  Deleting unused imports is smaller than trying to fix them."
                                ),
                                'suggested': "DELETE this line (smallest change for unused imports)"
                            }
                        elif modules_with_usage and not modules_to_delete:
                            # ALL modules have usage - provide exact replacement imports
                            new_imports = []
                            for mod_name, mod_path, classes in modules_with_usage:
                                class_list = ', '.join(classes[:3])
                                new_imports.append(f"from {mod_path} import {class_list}")
                            
                            suggested_imports = '\n'.join(new_imports)
                            return {
                                'valid': False,
                                'message': (
                                    f"INVALID: Cannot import modules this way.\n"
                                    f"  These are MODULES, not classes.\n\n"
                                    f"  Your code uses classes from these modules.\n\n"
                                    f"  SMALLEST CHANGE: Replace with proper imports:\n"
                                    f"    {suggested_imports}"
                                ),
                                'suggested': f"{suggested_imports} (smallest fix - just change the import)"
                            }
                        else:
                            # MIXED: some modules used, some not
                            unused = ', '.join(modules_to_delete)
                            used_info = []
                            for mod_name, mod_path, classes in modules_with_usage:
                                class_list = ', '.join(classes[:3])
                                used_info.append(f"from {mod_path} import {class_list}")
                            
                            suggested_imports = '\n'.join(used_info)
                            return {
                                'valid': False,
                                'message': (
                                    f"INVALID: Cannot import modules this way.\n"
                                    f"  These are MODULES, not classes.\n\n"
                                    f"  NOT USED (delete): {unused}\n"
                                    f"  USED (need proper import):\n\n"
                                    f"  SMALLEST CHANGE: Replace this line with:\n"
                                    f"    {suggested_imports}\n"
                                    f"  (removes unused {unused}, fixes the rest)"
                                ),
                                'suggested': f"{suggested_imports} (smallest fix)"
                            }
                    
                    # After handling module imports, check if first item is a regular class
                    class_name = class_names[0]
                    info = self.get_class_info(class_name)
                    
                    if not info:
                        return {
                            'valid': False,
                            'message': (
                                f"INVALID: Class '{class_name}' not found in VTK API.\n"
                                f"  This class doesn't exist - likely a hallucination or typo.\n"
                                f"  SMALLEST CHANGE: Remove or replace with a real VTK class name"
                            ),
                            'suggested': "DELETE or replace with valid class name (smallest fix)"
                        }
                    
                    correct_module = info['module']
                    module_part = parts[0].replace('from', '').strip()
                    
                    if module_part == correct_module:
                        return {
                            'valid': True,
                            'message': f"Import is correct",
                            'suggested': None
                        }
                    else:
                        # Special handling for vtkmodules.all
                        if module_part == 'vtkmodules.all':
                            # Importing classes from vtkmodules.all is VALID (though not best practice)
                            # Only reject if it's wrong entirely
                            return {
                                'valid': True,
                                'message': f"Import is valid (though importing from specific module {correct_module} is preferred)",
                                'suggested': None
                            }
                        
                        # Regular incorrect module error
                        suggested = f"from {correct_module} import {class_part}"
                        return {
                            'valid': False,
                            'message': (
                                f"import: INVALID: Incorrect module.\n"
                                f"  '{class_name}' is in '{correct_module}', not '{module_part}'\n\n"
                                f"  REPLACE THIS EXACT LINE:\n"
                                f"    {import_statement.strip()}\n"
                                f"  WITH:\n"
                                f"    {suggested}\n\n"
                                f"  REQUIRED: Change module from '{module_part}' to '{correct_module}'"
                            ),
                            'suggested': suggested
                        }
        
        return {
            'valid': False,
            'message': "Could not parse import statement",
            'suggested': None
        }
    
    def get_method_info(self, class_name: str, method_name: str) -> Optional[Dict[str, str]]:
        """Get information about a specific method of a class"""
        info = self.get_class_info(class_name)
        if not info:
            return None
        
        # Check if structured_docs exists (raw format)
        metadata = info.get('metadata', {})
        structured_docs = metadata.get('structured_docs', {})
        
        if structured_docs:
            # Raw format: use structured_docs
            sections = structured_docs.get('sections', {})
            
            # Check all method sections
            for section_name, section_data in sections.items():
                if 'methods' in section_data:
                    methods = section_data['methods']
                    if method_name in methods:
                        return {
                            'class_name': class_name,
                            'method_name': method_name,
                            'content': methods[method_name],
                            'section': section_name
                        }
        
        # Fallback: search in content (for chunked format or if structured_docs missing)
        content = info.get('content', '')
        lines = content.split('\n')
        in_methods = False
        method_lines = []
        
        for line in lines:
            if '## |  Methods defined here:' in line:
                in_methods = True
                continue
            
            if in_methods:
                if line.startswith('###') and method_name not in line:
                    # Next method, stop
                    break
                method_lines.append(line)
        
        if method_lines:
            return {
                'class_name': class_name,
                'method_name': method_name,
                'content': '\n'.join(method_lines)
            }
        
        return None


class VTKAPIMCPServer:
    """MCP Server for VTK API access"""
    
    def __init__(self, api_docs_path: Path):
        self.api_index = VTKAPIIndex(api_docs_path)
        self.server = Server("vtk-api")
        self._setup_tools()
    
    def _setup_tools(self):
        """Register all MCP tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="vtk_get_class_info",
                    description="Get complete information about a VTK class including module path, description, and methods",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "class_name": {
                                "type": "string",
                                "description": "VTK class name (e.g., 'vtkPolyDataMapper')"
                            }
                        },
                        "required": ["class_name"]
                    }
                ),
                Tool(
                    name="vtk_search_classes",
                    description="Search for VTK classes by name or keyword",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search term (e.g., 'reader', 'mapper', 'actor')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="vtk_get_module_classes",
                    description="List all VTK classes in a specific module",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "module": {
                                "type": "string",
                                "description": "Module name (e.g., 'vtkmodules.vtkRenderingCore')"
                            }
                        },
                        "required": ["module"]
                    }
                ),
                Tool(
                    name="vtk_validate_import",
                    description="Validate if a VTK import statement is correct and suggest corrections",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "import_statement": {
                                "type": "string",
                                "description": "Python import statement to validate"
                            }
                        },
                        "required": ["import_statement"]
                    }
                ),
                Tool(
                    name="vtk_get_method_info",
                    description="Get documentation for a specific method of a VTK class",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "class_name": {
                                "type": "string",
                                "description": "VTK class name"
                            },
                            "method_name": {
                                "type": "string",
                                "description": "Method name"
                            }
                        },
                        "required": ["class_name", "method_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls"""
            
            if name == "vtk_get_class_info":
                class_name = arguments["class_name"]
                info = self.api_index.get_class_info(class_name)
                
                if info:
                    result = {
                        "class_name": info['class_name'],
                        "module": info['module'],
                        "content_preview": info['content'][:500] + "..."
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                else:
                    return [TextContent(type="text", text=f"Class '{class_name}' not found in VTK API")]
            
            elif name == "vtk_search_classes":
                query = arguments["query"]
                limit = arguments.get("limit", 10)
                results = self.api_index.search_classes(query, limit)
                return [TextContent(type="text", text=json.dumps(results, indent=2))]
            
            elif name == "vtk_get_module_classes":
                module = arguments["module"]
                classes = self.api_index.get_module_classes(module)
                result = {
                    "module": module,
                    "classes": classes,
                    "count": len(classes)
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "vtk_validate_import":
                import_statement = arguments["import_statement"]
                result = self.api_index.validate_import(import_statement)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "vtk_get_method_info":
                class_name = arguments["class_name"]
                method_name = arguments["method_name"]
                info = self.api_index.get_method_info(class_name, method_name)
                
                if info:
                    return [TextContent(type="text", text=json.dumps(info, indent=2))]
                else:
                    return [TextContent(type="text", text=f"Method '{method_name}' not found in class '{class_name}'")]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            logger.info("VTK API MCP Server starting...")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VTK API MCP Server")
    parser.add_argument(
        "--api-docs",
        type=Path,
        default=Path("../data/processed/chunked-api-docs.jsonl"),
        help="Path to chunked API docs file"
    )
    
    args = parser.parse_args()
    
    server = VTKAPIServer(args.api_docs)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
