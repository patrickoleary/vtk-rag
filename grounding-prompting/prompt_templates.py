#!/usr/bin/env python3
"""
Prompt Templates for VTK RAG - Production System

Provides JSON-based structured prompts for:
- Sequential Pipeline: Complex multi-step queries
- Simple Queries: Single-task CODE, API, EXPLANATION queries

All prompts are designed for clean JSON input/output with LLM.
"""

from typing import Dict, Any


class VTKPromptTemplate:
    """
    Prompt template generator for VTK RAG
    
    Provides structured instructions for:
    
    SEQUENTIAL PIPELINE (Complex queries):
    1. Query decomposition (complex → steps)
    2. Per-step code generation (step + docs → code)
    
    SIMPLE QUERIES (Single-task):
    3. Code generation (query + docs → code)
    4. API lookup (query + docs → API explanation)
    5. Explanation (query + docs → concept explanation)
    6. Image-to-code (image + docs → code that produces it)
    7. Code-to-image (code + docs → expected output image)
    8. Data-to-code (data file + docs → code to process it)
    9. Code-to-data (code + docs → example data files)
    10. Clarifying questions (ambiguous query → question)
    
    All methods return Dict suitable for JSON serialization.
    """
    
    def get_decomposition_instructions(self) -> Dict:
        """
        Get instructions for query decomposition (JSON-based)
        
        Used by sequential_pipeline.py to break complex queries into steps.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with role, task, think_through, requirements, etc.
        """
        return {
            "role": "You are an expert VTK application developer with deep knowledge of the VTK pipeline and Python scientific computing.",
            "task": "Analyze this query and break it into logical tasks. Return your response as JSON with the EXACT structure shown in the example.",
            "think_through": [
                "What is the user actually trying to accomplish? (visualization? computation? file I/O?)",
                "What specific libraries or tools are mentioned? (pandas? numpy? specific VTK classes?)",
                "What specific data files are mentioned? (CSV files, STL files, etc.)",
                "Does this require visualization, or just computation/output?",
                "What are the 3-8 concrete steps needed to solve this?"
            ],
            "output_format_example": {
                "understanding": "Brief summary of what user wants",
                "requires_visualization": True,
                "libraries_needed": ["vtk", "pandas"],
                "data_files": ["example.csv"],
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Read example.csv with pandas and extract columns",
                        "search_query": "pandas read CSV file extract columns",
                        "focus": "data_io"
                    }
                ]
            },
            "requirements": [
                "Return JSON matching the exact structure in output_format_example",
                "DO NOT force visualization if not needed",
                "DO NOT ignore specific libraries mentioned in the query (pandas, numpy, etc.)",
                "DO NOT ignore specific data files mentioned in the query (preserve exact filenames)",
                "DO NOT use generic queries like 'VTK create mapper' - be specific",
                "DO NOT write code - only analyze and plan",
                "Include specific filenames and column names in 'description', but use generic patterns in 'search_query'"
            ],
            "focus_areas": "data_io, data_processing, geometry, filtering, transformation, visualization, rendering, utility"
        }
    
    def get_generation_instructions(self) -> Dict:
        """
        Get instructions for per-step code generation (JSON-based)
        
        Used by sequential_pipeline.py to generate code for each step.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with task, requirements, output_format
        """
        return {
            "task": "Generate code for the current step only",
            "requirements": [
                # Code structure requirements
                "Generate ONLY the Python code statements needed for this step",
                "Put ALL import statements in the 'imports' array field",
                "Put ONLY executable code (no imports) in the 'code' field",
                "Continue from the existing code in previous_steps",
                
                # GROUNDING: Anti-hallucination and citation requirements
                "Review the documentation for relevant examples",
                "DO NOT hallucinate - only use patterns from documentation if they apply",
                "Stay faithful to the original problem requirements",
                "If the step mentions specific libraries/files, use them",
                "Cite documentation using the index number [N]"
            ],
            "output_format": "Return JSON with: step_number, understanding, imports (list of import statements ONLY), code (executable Python code WITHOUT imports), citations (list of doc indices)"
        }
    
    # === SIMPLE QUERY INSTRUCTIONS (NEW) ===
    
    def get_code_generation_instructions(self) -> Dict:
        """
        Get instructions for simple CODE query generation (JSON-based)
        
        Used for single-task code generation queries.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK code generation assistant specialized in creating Python visualization code.",
            "task": "Generate complete, working Python code to solve the user's VTK problem. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "code",
                "code": "import vtkmodules.vtkRenderingOpenGL2\nfrom vtkmodules.vtkFiltersSources import vtkCylinderSource\nfrom vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow\n\ncylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)\n\nmapper = vtkPolyDataMapper()\nmapper.SetInputConnection(cylinder.GetOutputPort())\n\nactor = vtkActor()\nactor.SetMapper(mapper)\n\nrenderer = vtkRenderer()\nrenderer.AddActor(actor)\n\nrender_window = vtkRenderWindow()\nrender_window.AddRenderer(renderer)\nrender_window.SetSize(800, 600)\nrender_window.Render()",
                "explanation": "This code creates and renders a 3D cylinder [1]. First, vtkRenderingOpenGL2 is imported to initialize the OpenGL rendering backend. Then vtkCylinderSource creates the cylinder geometry with a radius of 5.0 units. The cylinder is connected to a vtkPolyDataMapper and vtkActor for rendering. Finally, a vtkRenderer and vtkRenderWindow are set up to display the result.",
                "vtk_classes_used": ["vtkCylinderSource", "vtkPolyDataMapper", "vtkActor", "vtkRenderer", "vtkRenderWindow"],
                "citations": [
                    {"number": 1, "reason": "cylinder creation pattern from documentation"}
                ],
                "data_files_needed": [],
                "requires_baseline": True,
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names (case-sensitive): response_type, content_type, code, explanation, vtk_classes_used, citations, data_files_needed, requires_baseline, confidence",
                "CRITICAL: For ANY code that uses vtkRenderWindow, vtkRenderer, or any rendering classes, you MUST start with: import vtkmodules.vtkRenderingOpenGL2",
                "This OpenGL2 import MUST be the very first import before any other VTK imports",
                "Without vtkRenderingOpenGL2, rendering code will crash with segmentation faults",
                "IMPORTANT: For VTK scalar type constants (VTK_UNSIGNED_CHAR, VTK_FLOAT, VTK_INT, etc.), use their numeric values directly instead of importing them",
                "VTK_UNSIGNED_CHAR=3, VTK_CHAR=2, VTK_SHORT=4, VTK_UNSIGNED_SHORT=5, VTK_INT=6, VTK_UNSIGNED_INT=7, VTK_FLOAT=10, VTK_DOUBLE=11",
                "Example: Use image_data.AllocateScalars(3, 1) instead of importing VTK_UNSIGNED_CHAR",
                "Put complete working code in the 'code' field",
                "Provide clear explanation of what the code does in the 'explanation' field",
                "List all VTK classes used in 'vtk_classes_used' array",
                "Cite sources using [N] notation in the explanation and include citation details",
                "If code requires data files, list them in 'data_files_needed'",
                "Set 'requires_baseline' to true if retrieved examples show expected output images",
                "Set 'confidence' to: 'high' if documentation is clear, 'medium' if some uncertainty, 'low' if documentation is insufficient"
            ],
            "grounding": [
                "Review ALL provided documentation chunks carefully",
                "Only use VTK classes and methods that appear in the documentation",
                "Do NOT hallucinate methods, parameters, or class names",
                "If the documentation shows data files, mention them in explanation and data_files_needed",
                "If the documentation shows expected output images, set requires_baseline=true",
                "If documentation is insufficient for a confident answer, set confidence='low' and explain why in the explanation"
            ]
        }
    
    def get_api_lookup_instructions(self) -> Dict:
        """
        Get instructions for API/documentation queries (JSON-based)
        
        Used for queries about VTK API methods, classes, and parameters.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK API documentation assistant specialized in explaining VTK classes and methods.",
            "task": "Explain the requested VTK API methods and classes clearly. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "api",
                "explanation": "vtkActor provides these key methods [1]:\n\n- SetMapper(mapper): Assigns a vtkPolyDataMapper to the actor\n- GetProperty(): Returns a vtkProperty object for customizing appearance\n- SetPosition(x, y, z): Sets the actor's position in 3D space\n- SetScale(x, y, z): Sets scaling factors along each axis\n- RotateX/Y/Z(angle): Rotates the actor around the specified axis",
                "api_references": [
                    {
                        "class_name": "vtkActor",
                        "methods_discussed": ["SetMapper", "GetProperty", "SetPosition", "SetScale", "RotateX", "RotateY", "RotateZ"]
                    }
                ],
                "citations": [
                    {"number": 1, "reason": "vtkActor API documentation"}
                ],
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, explanation, api_references, citations, confidence",
                "Provide clear, detailed explanation of API methods and their purposes",
                "List all classes discussed in 'api_references' with their methods",
                "Include parameter information and return types when available",
                "Cite API documentation sources using [N] notation",
                "Set 'confidence' based on documentation completeness"
            ],
            "grounding": [
                "Only describe methods and classes that appear in the provided API documentation",
                "Do NOT invent method signatures, parameters, or return types",
                "If parameter types or return types are not in documentation, state 'not specified in documentation'",
                "If documentation is incomplete, set confidence='medium' or 'low' and note what's missing"
            ]
        }
    
    def get_explanation_instructions(self) -> Dict:
        """
        Get instructions for concept/explanation queries (JSON-based)
        
        Used for queries asking about VTK concepts, workflows, and architecture.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK concepts explanation assistant specialized in teaching VTK workflows and architecture.",
            "task": "Explain the requested VTK concept clearly with examples. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "explanation",
                "explanation": "vtkCylinderSource is a VTK class that generates 3D cylinder geometry [1]. It's a procedural source that creates polygonal data representing a cylinder.\n\nKey characteristics:\n- Generates cylinder centered at origin by default\n- Oriented along Y-axis\n- Can control radius, height, and resolution\n- Output is vtkPolyData [1]\n\nCommon uses:\n- Creating geometric primitives for visualization [2]\n- Testing rendering pipelines\n- Building 3D scenes with basic shapes",
                "key_concepts": ["procedural geometry", "polygonal data", "VTK pipeline", "geometric sources"],
                "citations": [
                    {"number": 1, "reason": "vtkCylinderSource documentation"},
                    {"number": 2, "reason": "geometric primitives examples"}
                ],
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, explanation, key_concepts, citations, confidence",
                "Provide clear, educational explanation suitable for learning",
                "Break down complex concepts into understandable parts",
                "Include examples from documentation when available",
                "List main concepts covered in 'key_concepts' array",
                "Cite information sources using [N] notation"
            ],
            "grounding": [
                "Base explanation entirely on the provided documentation",
                "Do NOT introduce concepts, workflows, or examples not present in documentation",
                "If documentation doesn't cover a requested aspect, explicitly state this",
                "Use specific examples from the documentation to illustrate concepts",
                "If documentation is insufficient for complete explanation, set confidence='low' and note what's missing"
            ]
        }
    
    def get_image_to_code_instructions(self) -> Dict:
        """
        Get instructions for image-to-code queries (JSON-based)
        
        🔮 FUTURE CAPABILITY - Requires multimodal retrieval (CLIP/BLIP-2)
        
        User provides an image and asks "what code produces this output?"
        Returns structured instructions dict for JSON prompt.
        
        NOTE: Not currently implemented. Requires:
        - Image embeddings in Qdrant
        - CLIP/BLIP-2 for image→text or image→embedding
        - Multimodal search capability
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK reverse engineering assistant specialized in analyzing visualization outputs and determining the code that produced them.",
            "task": "Analyze the provided image and identify VTK code patterns from the documentation that could produce similar output. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "code",
                "code": "from vtkmodules.vtkFiltersSources import vtkCylinderSource\nfrom vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow\n\ncylinder = vtkCylinderSource()\ncylinder.SetRadius(5.0)\ncylinder.SetHeight(10.0)\n\nmapper = vtkPolyDataMapper()\nmapper.SetInputConnection(cylinder.GetOutputPort())\n\nactor = vtkActor()\nactor.SetMapper(mapper)\n\nrenderer = vtkRenderer()\nrenderer.AddActor(actor)\n\nrenderWindow = vtkRenderWindow()\nrenderWindow.AddRenderer(renderer)",
                "explanation": "Based on the image showing a 3D cylinder, this code uses vtkCylinderSource to create the geometry [1]. The rendering pipeline connects the cylinder through a mapper to an actor, which is added to a renderer [2]. The image characteristics (solid color, basic shading) suggest default rendering properties.",
                "image_analysis": "The image shows a 3D cylinder with smooth shading, suggesting polygonal geometry with sufficient resolution. The orientation appears to be along the Y-axis (default for vtkCylinderSource). No custom colors or textures are visible.",
                "vtk_classes_used": ["vtkCylinderSource", "vtkPolyDataMapper", "vtkActor", "vtkRenderer", "vtkRenderWindow"],
                "citations": [
                    {"number": 1, "reason": "cylinder source example from documentation"},
                    {"number": 2, "reason": "basic rendering pipeline pattern"}
                ],
                "confidence": "high",
                "matching_examples": ["CylinderExample.py"]
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, code, explanation, image_analysis, vtk_classes_used, citations, confidence, matching_examples",
                "Analyze the image carefully: geometry type, rendering style, colors, camera angle",
                "Provide code that would produce visually similar output",
                "In 'image_analysis' field, describe what you observe in the image",
                "In 'explanation' field, explain how the code produces the observed output",
                "List any matching examples from documentation in 'matching_examples'",
                "Set confidence based on how well documentation matches the image",
                "IMPORTANT: If code uses vtkRenderWindow and calls .Render(), you MUST include: import vtkmodules.vtkRenderingOpenGL2 at the top of the code to initialize the OpenGL rendering backend"
            ],
            "grounding": [
                "Search documentation for examples with similar visual output",
                "Only use VTK classes and patterns that appear in the documentation",
                "If image shows features not covered in documentation, note this and set confidence='low'",
                "Prefer examples with baseline images that match the provided image",
                "If multiple examples could produce similar output, mention the closest match"
            ]
        }
    
    def get_code_to_image_instructions(self) -> Dict:
        """
        Get instructions for code-to-image queries (JSON-based)
        
        ✅ IMPLEMENTED - Extracts baseline images from retrieved chunks
        
        User provides code and asks "what does this produce?" or "show me the expected output"
        Returns structured instructions dict for JSON prompt.
        
        Implementation: Searches for similar code examples in documentation and returns
        baseline images from matching chunks. Does NOT execute code.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK visualization assistant specialized in explaining what code produces and finding matching output images from the documentation.",
            "task": "Analyze the provided code, find matching examples in the documentation with baseline images, and describe the expected visual output. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "image",
                "explanation": "This code creates a 3D cylinder using vtkCylinderSource and renders it with basic properties [1]. The expected output is a solid-colored cylinder oriented along the Y-axis with smooth shading.",
                "visual_description": "The output will show a 3D cylinder with:\n- Smooth polygonal surface\n- Default white/gray color\n- Basic lighting and shading\n- Y-axis orientation (vertical)\n- Centered at origin",
                "baseline_images": [
                    {
                        "url": "https://vtk.org/files/ExternalData/Testing/Data/Baseline/Cylinder.png",
                        "source": "CylinderExample.py",
                        "similarity": "exact"
                    }
                ],
                "vtk_classes_used": ["vtkCylinderSource", "vtkPolyDataMapper", "vtkActor"],
                "citations": [
                    {"number": 1, "reason": "cylinder rendering example with baseline image"}
                ],
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, explanation, visual_description, baseline_images, vtk_classes_used, citations, confidence",
                "Analyze the code to understand what visual output it produces",
                "Search documentation for examples with similar code patterns",
                "If examples have baseline images, include them in 'baseline_images' array",
                "Describe expected visual output in 'visual_description' field",
                "Set 'similarity' to: 'exact' if code matches example, 'similar' if close, 'approximate' if general pattern"
            ],
            "grounding": [
                "Search documentation for examples with matching VTK class usage",
                "Only suggest baseline images from examples in the documentation",
                "If no baseline images are found in documentation, return empty baseline_images array and describe expected output",
                "Do NOT hallucinate or invent image URLs",
                "If code doesn't match any documented examples, set confidence='low' and describe expected output generically"
            ]
        }
    
    def get_data_to_code_instructions(self) -> Dict:
        """
        Get instructions for data-to-code queries (JSON-based)
        
        ✅ IMPLEMENTED - Shows multiple visualization techniques for data files
        
        User provides/describes data file and asks "what can I do with this?" or "how do I visualize this?"
        Returns structured instructions dict for JSON prompt.
        
        Implementation: Searches for examples with matching file types, groups by category,
        and suggests multiple techniques. Returns working code for one technique plus
        alternative approaches.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK data processing assistant specialized in suggesting multiple visualization and processing techniques for different data types.",
            "task": "Analyze the data file type, review matching examples grouped by technique category, and suggest multiple approaches. Provide working code for the most common technique, and describe alternatives. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "code",
                "code": "import pandas as pd\nfrom vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray\nfrom vtkmodules.vtkCommonDataModel import vtkPolyData\n\n# Read CSV data\ndf = pd.read_csv('points.csv')\n\n# Create VTK points\npoints = vtkPoints()\nfor _, row in df.iterrows():\n    points.InsertNextPoint(row['x'], row['y'], row['z'])\n\n# Create polydata\npolydata = vtkPolyData()\npolydata.SetPoints(points)",
                "explanation": "This code reads a CSV file with x, y, z columns and converts it to VTK points [1]. The pandas library loads the CSV, then vtkPoints is populated from the dataframe. This creates a point cloud suitable for visualization [2].",
                "data_analysis": "CSV file with numeric columns 'x', 'y', 'z' representing 3D point coordinates. This is suitable for point cloud visualization, scatter plots, or surface reconstruction.",
                "suggested_techniques": [
                    "Point cloud visualization with vtkPolyData",
                    "3D scatter plot with vtkGlyph3D",
                    "Surface reconstruction with vtkDelaunay3D",
                    "Statistical analysis with vtkDescriptiveStatistics"
                ],
                "vtk_classes_used": ["vtkPoints", "vtkFloatArray", "vtkPolyData"],
                "data_files_used": ["points.csv"],
                "alternative_approaches": [
                    {
                        "technique": "3D Scatter Plot",
                        "description": "Use vtkGlyph3D to show spheres at each point, better for seeing individual data points",
                        "vtk_classes": ["vtkGlyph3D", "vtkSphereSource"],
                        "complexity": "moderate"
                    },
                    {
                        "technique": "Surface Reconstruction",
                        "description": "Use vtkDelaunay3D to create a surface mesh from the point cloud",
                        "vtk_classes": ["vtkDelaunay3D"],
                        "complexity": "advanced"
                    }
                ],
                "citations": [
                    {"number": 1, "reason": "CSV to VTK points conversion pattern"},
                    {"number": 2, "reason": "point cloud visualization example"}
                ],
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, code, explanation, data_analysis, suggested_techniques, vtk_classes_used, data_files_used, alternative_approaches, citations, confidence",
                "Analyze the data file type (CSV, STL, VTI, VTP, etc.)",
                "Provide complete working code for the MOST COMMON technique in 'code' field",
                "List 3-5 technique names in 'suggested_techniques' array",
                "Provide detailed alternative approaches in 'alternative_approaches' array with descriptions and VTK classes",
                "In 'data_analysis' field, describe what the data represents and what it's suitable for",
                "IMPORTANT: If code uses vtkRenderWindow and calls .Render(), you MUST include: import vtkmodules.vtkRenderingOpenGL2 at the top of the code to initialize the OpenGL rendering backend"
            ],
            "grounding": [
                "Search documentation for examples using similar data file types",
                "Only suggest VTK classes and techniques that appear in the documentation",
                "If documentation has examples with matching file types, cite them",
                "If data type is not covered in documentation, set confidence='low' and explain limitations",
                "Prefer examples that show complete data loading and visualization pipelines"
            ]
        }
    
    def get_code_to_data_instructions(self) -> Dict:
        """
        Get instructions for code-to-data queries (JSON-based)
        
        ✅ IMPLEMENTED - Finds example data files from documentation metadata
        
        User provides code and asks "do you have example data for this?" or "what data files can I use?"
        Returns structured instructions dict for JSON prompt.
        
        Implementation: Parses code to identify reader type (e.g., vtkSTLReader), searches
        for examples with matching file types, and extracts data file info (filename, URL,
        size) from chunk metadata. Returns list of downloadable files.
        
        Returns:
            Dict with role, task, output_format, requirements, grounding
        """
        return {
            "role": "You are a VTK data resource assistant specialized in finding appropriate example data files for code from the VTK documentation and testing data.",
            "task": "Analyze the provided code to determine what data format it needs, find matching example data files from the documentation, and provide download information. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "answer",
                "content_type": "data",
                "explanation": "This code reads and visualizes STL mesh files [1]. Several example STL files are available in the VTK testing data that would work with this code.",
                "data_files": [
                    {
                        "filename": "42400-IDGH.stl",
                        "description": "STL mesh file - tooth model",
                        "source": "STLReader example",
                        "download_url": "https://vtk.org/files/ExternalData/Testing/Data/42400-IDGH.stl",
                        "file_type": "STL",
                        "size_info": "~2MB, 40K triangles"
                    },
                    {
                        "filename": "sphere.stl",
                        "description": "Simple sphere mesh for testing",
                        "source": "Basic STL example",
                        "download_url": "https://vtk.org/files/ExternalData/Testing/Data/sphere.stl",
                        "file_type": "STL",
                        "size_info": "~100KB, 2K triangles"
                    }
                ],
                "code_requirements": "Code expects STL file with triangular mesh data. Any valid STL file (ASCII or binary) will work.",
                "vtk_classes_used": ["vtkSTLReader", "vtkPolyDataMapper"],
                "citations": [
                    {"number": 1, "reason": "STL reader example with data files"}
                ],
                "confidence": "high"
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, content_type, explanation, data_files, code_requirements, vtk_classes_used, citations, confidence",
                "Analyze code to determine what data format it expects",
                "Search documentation for examples with matching file types",
                "Only include data files that are actually available in the documentation/testing data",
                "Provide download URLs for each data file",
                "In 'code_requirements' field, describe what the code expects from data files"
            ],
            "grounding": [
                "Only list data files that appear in the documentation examples",
                "Do NOT hallucinate or invent data file names or URLs",
                "If no matching data files are found, return empty data_files array and set confidence='low'",
                "Prefer data files from examples that use similar VTK classes",
                "Include file size and complexity information if available in documentation"
            ]
        }
    
    def get_clarifying_question_instructions(self) -> Dict:
        """
        Get instructions for clarifying questions (JSON-based)
        
        Used when user query is too ambiguous to answer directly.
        Returns structured instructions dict for JSON prompt.
        
        Returns:
            Dict with role, task, output_format, requirements
        """
        return {
            "role": "You are a VTK assistant helping users clarify their needs.",
            "task": "When the query is ambiguous or could have multiple interpretations, ask a clarifying question. Return your response as valid JSON.",
            "output_format_example": {
                "response_type": "clarifying_question",
                "question": "What type of visualization do you need for your cylinder?",
                "options": [
                    "Basic solid cylinder rendering",
                    "Cylinder with custom colors and properties",
                    "Multiple cylinders in a scene",
                    "Animated rotating cylinder",
                    "Cylinder as part of a larger pipeline"
                ],
                "reason": "The query 'visualize cylinder' is too vague - there are multiple ways to visualize cylinders in VTK, each requiring different approaches."
            },
            "requirements": [
                "Return ONLY valid JSON matching the EXACT structure in output_format_example",
                "Use the EXACT field names: response_type, question, options, reason",
                "Ask ONE clear, specific question",
                "Provide 3-5 distinct, actionable options",
                "Make options specific enough to guide next steps",
                "Explain why clarification is needed in 'reason' field"
            ],
            "when_to_use": [
                "Query is too vague (e.g., 'how to visualize data' without context)",
                "Multiple valid interpretations exist",
                "User hasn't specified required details (data type, visualization type, etc.)",
                "Documentation shows multiple approaches and it's unclear which the user wants"
            ]
        }
    
    def get_code_explanation_generation_instructions(self) -> Dict[str, Any]:
        """
        Instructions for generating detailed code explanations
        
        Returns:
            Dict with role, task, requirements for explanation generation
        """
        return {
            "role": "You are a VTK expert educator creating clear, detailed explanations of VTK code.",
            "task": "Generate a comprehensive explanation of the provided VTK code that helps users understand what it does and how it works.",
            "requirements": [
                "Explain the overall purpose of the code",
                "Describe each major VTK class used and its role",
                "Explain the data flow through the VTK pipeline",
                "Highlight important method calls and what they configure",
                "Mention any common patterns or best practices demonstrated",
                "Use clear, educational language suitable for intermediate Python/VTK users",
                "Include inline citations [N] referencing the documentation chunks used",
                "Return structured JSON with ExplanationEnrichmentOutput schema",
                "Set confidence level based on code clarity and documentation quality"
            ],
            "output_format": "JSON matching ExplanationEnrichmentOutput schema"
        }
    
    def get_explanation_improvement_instructions(self) -> Dict[str, Any]:
        """
        Instructions for improving existing code explanations
        
        Returns:
            Dict with role, task, requirements for explanation improvement
        """
        return {
            "role": "You are a VTK expert educator improving code explanations to be more detailed and educational.",
            "task": "Enhance the existing explanation by adding more detail, clarity, and educational context while preserving the original intent.",
            "requirements": [
                "Keep the core message of the original explanation",
                "Add more detail about VTK classes and their purposes",
                "Explain the pipeline/dataflow more clearly if applicable",
                "Add information about important method calls and parameters",
                "Include educational context (when to use this pattern, alternatives, etc.)",
                "Maintain or improve clarity - don't make it unnecessarily verbose",
                "Use inline citations [N] for any new information from documentation",
                "Return structured JSON with ExplanationEnrichmentOutput schema",
                "Set confidence level based on improvement quality"
            ],
            "output_format": "JSON matching ExplanationEnrichmentOutput schema"
        }
