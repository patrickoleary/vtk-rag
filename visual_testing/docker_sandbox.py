#!/usr/bin/env python3
"""
Docker Sandbox for VTK Code Execution

Executes VTK code in isolated Docker container with:
- Resource limits (memory, CPU, timeout)
- Offscreen rendering
- Image capture
- Network isolation
"""

import docker
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DockerSandbox:
    """
    Execute VTK code in secure Docker sandbox
    
    Features:
    - Isolated execution (cannot access host filesystem)
    - Resource limits (memory, CPU, timeout)
    - Offscreen rendering with Xvfb
    - Image capture from render window
    - Automatic cleanup
    """
    
    def __init__(
        self,
        image_name: str = "vtk-visual-testing",
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU
        timeout: int = 30,  # seconds
        network_disabled: bool = True
    ):
        """
        Initialize Docker sandbox
        
        Args:
            image_name: Docker image name
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_quota: CPU quota in microseconds (100000 = 1 CPU)
            timeout: Execution timeout in seconds
            network_disabled: Disable network access
        """
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.timeout = timeout
        self.network_disabled = network_disabled
        
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def execute_code(
        self,
        code: str,
        test_name: str,
        data_files: Optional[Dict[str, bytes]] = None
    ) -> Dict:
        """
        Execute VTK code in Docker sandbox
        
        Args:
            code: Python code to execute
            test_name: Name for this test (used for output files)
            data_files: Optional dict of filename -> content for data files
            
        Returns:
            Dict with:
                - success: bool
                - output: str (stdout/stderr)
                - output_image: bytes (PNG image if rendered)
                - execution_time: float
                - error: str (if failed)
        """
        start_time = time.time()
        
        # Create temporary directory for code and output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write code to file
            code_file = tmpdir_path / "test_code.py"
            
            # Rewrite data paths for Docker environment
            code_rewritten = self._rewrite_data_paths(code)
            
            # Wrap code to capture output image
            wrapped_code = self._wrap_code_for_capture(code_rewritten, test_name)
            code_file.write_text(wrapped_code)
            
            # Write data files if provided
            if data_files:
                for filename, content in data_files.items():
                    (tmpdir_path / filename).write_bytes(content)
            
            # Create output directory
            output_dir = tmpdir_path / "output"
            output_dir.mkdir()
            
            try:
                # Run container
                result = self._run_container(tmpdir_path, output_dir)
                
                execution_time = time.time() - start_time
                
                # Check for output image
                output_image = None
                image_path = output_dir / f"{test_name}.png"
                if image_path.exists():
                    output_image = image_path.read_bytes()
                
                return {
                    'success': result['success'],
                    'output': result['output'],
                    'output_image': output_image,
                    'execution_time': execution_time,
                    'error': result.get('error')
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Docker execution failed: {e}")
                return {
                    'success': False,
                    'output': '',
                    'output_image': None,
                    'execution_time': execution_time,
                    'error': str(e)
                }
    
    def _rewrite_data_paths(self, code: str) -> str:
        """
        Rewrite data file paths for Docker environment
        
        Converts bare filenames and data/ paths to /data/ paths:
        - 'LakeGininderra.csv' -> '/data/LakeGininderra.csv'
        - 'data/raw/file.csv' -> '/data/file.csv'
        - "headMesh.stl" -> "/data/headMesh.stl"
        
        Args:
            code: Python code with data file references
            
        Returns:
            Code with rewritten paths for Docker /data mount
        """
        import re
        
        # Known VTK data file extensions
        data_extensions = r'\.(stl|csv|vtu|vtp|vti|vts|vtr|vtk|obj|ply|e|slc|bin|kmz|xml|pvd|pvtu|pvti)'
        
        # Pattern: Match bare filenames with data extensions
        # Matches: 'file.stl' or "file.csv" but not '/data/file.stl' (already absolute)
        pattern_bare = rf'''(['"])(?![/\\])([^'"]+{data_extensions})\1'''
        
        def replace_bare(match):
            quote = match.group(1)
            filename = match.group(2)
            # If filename contains 'data/' prefix, strip it
            if 'data/' in filename or 'data\\' in filename:
                filename = filename.split('data/')[-1].split('data\\')[-1]
            return f'{quote}/data/{filename}{quote}'
        
        rewritten = re.sub(pattern_bare, replace_bare, code, flags=re.IGNORECASE)
        
        return rewritten
    
    def _wrap_code_for_capture(self, code: str, test_name: str) -> str:
        """
        Wrap VTK code to capture rendered output
        
        VTK 9.5.0+ with OSMesa handles offscreen rendering automatically.
        This wrapper just adds image capture at the end.
        """
        # Build wrapper - use format() to interpolate test_name
        wrapper = '''
import sys
import traceback

try:
    # Original code (VTK 9.5.0+ handles offscreen rendering automatically)
{code_indented}
    
    # Try to capture output if render window exists
    try:
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter, vtkRenderWindow
        
        # Find render window instance
        local_vars = dict(locals())
        render_window_found = None
        
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, vtkRenderWindow):
                render_window_found = var_value
                break
        
        if render_window_found:
            # Render to ensure everything is drawn (offscreen with OSMesa)
            render_window_found.Render()
            
            # Capture window to image using vtkWindowToImageFilter
            window_to_image = vtkWindowToImageFilter()
            window_to_image.SetInput(render_window_found)
            window_to_image.ReadFrontBufferOff()  # Use back buffer for offscreen
            window_to_image.Update()
            
            # Write PNG to mounted output directory
            writer = vtkPNGWriter()
            writer.SetFileName('/output/{output_file}')
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
            # Verify file was created
            import os
            if os.path.exists('/output/{output_file}'):
                file_size = os.path.getsize('/output/{output_file}')
                print(f"✓ Output image captured ({{file_size}} bytes)")
            else:
                print("✗ Image file not created")
        else:
            print("ℹ No render window found - no image captured")
            
    except Exception as capture_error:
        print(f"Warning: Could not capture output image: {{capture_error}}")
        traceback.print_exc()
    
    print("✓ Code executed successfully")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error executing code: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''.format(
            code_indented=self._indent_code(code, 4),
            output_file=f"{test_name}.png"
        )
        return wrapper
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))
    
    def _run_container(self, code_dir: Path, output_dir: Path) -> Dict:
        """
        Run Docker container with code
        
        Args:
            code_dir: Directory containing code file
            output_dir: Directory for output files
            
        Returns:
            Dict with success, output, error
        """
        try:
            # Prepare volumes - mount data directory for VTK data files
            volumes = {
                str(code_dir): {'bind': '/workspace', 'mode': 'ro'},
                str(output_dir): {'bind': '/output', 'mode': 'rw'}
            }
            
            # Mount evaluation data directory if it exists (for VTK example data files)
            # These are the actual STL, CSV, VTU files used by examples
            eval_data_dir = Path(__file__).parent.parent / 'evaluation' / 'data'
            if eval_data_dir.exists():
                volumes[str(eval_data_dir)] = {'bind': '/data', 'mode': 'ro'}
                logger.debug(f"Mounting data directory: {eval_data_dir}")
            
            # Run container with timeout using detach=True and wait
            # VTK 9.5+ handles offscreen rendering natively
            container = self.client.containers.run(
                self.image_name,
                command=["python3", "/workspace/test_code.py"],
                volumes=volumes,
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                network_disabled=self.network_disabled,
                remove=False,  # Don't auto-remove so we can get logs
                detach=True,
                stdout=True,
                stderr=True
            )
            
            # Wait for container to complete with timeout
            exit_code = container.wait(timeout=self.timeout)
            
            # Get output
            output = container.logs().decode('utf-8')
            
            # Remove container
            container.remove()
            
            # Check exit code
            if isinstance(exit_code, dict):
                exit_code = exit_code.get('StatusCode', 0)
            
            if exit_code != 0:
                return {
                    'success': False,
                    'output': output,
                    'error': f"Container exited with code {exit_code}"
                }
            
            return {
                'success': True,
                'output': output
            }
        
        except docker.errors.ContainerError as e:
            # Container exited with non-zero code
            output = e.stderr.decode('utf-8') if e.stderr else str(e)
            return {
                'success': False,
                'output': output,
                'error': f"Container exited with code {e.exit_status}"
            }
        
        except docker.errors.APIError as e:
            return {
                'success': False,
                'output': '',
                'error': f"Docker API error: {e}"
            }
        
        except Exception as e:
            # Handle timeout or other errors
            if 'container' in locals():
                try:
                    output = container.logs().decode('utf-8')
                    container.remove(force=True)
                except:
                    output = ''
            else:
                output = ''
            
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                error_msg = f"Execution timeout ({self.timeout}s)"
            
            return {
                'success': False,
                'output': output,
                'error': error_msg
            }
    
    def build_image(self, dockerfile_path: Path) -> bool:
        """
        Build Docker image from Dockerfile
        
        Args:
            dockerfile_path: Path to directory containing Dockerfile
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Building Docker image '{self.image_name}'...")
            
            image, build_logs = self.client.images.build(
                path=str(dockerfile_path),
                tag=self.image_name,
                rm=True,
                forcerm=True
            )
            
            # Print build logs
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            logger.info(f"✓ Image '{self.image_name}' built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return False
    
    def image_exists(self) -> bool:
        """Check if Docker image exists"""
        try:
            self.client.images.get(self.image_name)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking image: {e}")
            return False
    
    def cleanup(self):
        """Cleanup Docker resources"""
        try:
            self.client.close()
        except:
            pass
