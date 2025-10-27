#!/usr/bin/env python3
"""
Clean script - Remove all generated files and start fresh

Removes:
- Virtual environment (.venv)
- Processed corpus data
- Vector database indexes
- Visualizations
- Evaluation results
- Visual testing baselines
- Python cache files
- Docker images (with --docker flag)
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import sys

def clean(clean_docker=False):
    """Remove all generated files"""
    
    items_to_remove = [
        # Virtual environment
        ('.venv', 'Virtual environment'),
        ('venv', 'Old virtual environment (if exists)'),
        
        # Generated data
        ('data/processed', 'Processed corpus chunks'),
        ('data/visualizations', 'Analysis visualizations'),
        
        # Evaluation results
        ('evaluation/results', 'Evaluation results'),
        
        # Visual testing baselines
        ('tests/visual_testing/baselines', 'Visual testing baseline images'),
        
        # Vector databases
        ('chroma_db', 'ChromaDB database (legacy)'),
        ('indexes', 'Vector indexes'),
        ('.qdrant', 'Qdrant database'),
        
        # Python cache
        ('__pycache__', 'Python cache'),
        ('*/__pycache__', 'Module cache directories'),
        ('*.pyc', 'Compiled Python files'),
        
        # Logs
        ('*.log', 'Log files'),
    ]
    
    print("=" * 80)
    print("CLEAN - Starting Fresh")
    print("=" * 80)
    print()
    
    removed_count = 0
    
    for path_str, description in items_to_remove:
        path = Path(path_str)
        
        if '*' in path_str:
            # Handle glob patterns
            for match in Path('.').rglob(path.name):
                if match.exists():
                    if match.is_file():
                        match.unlink()
                        print(f"✓ Removed: {match} ({description})")
                        removed_count += 1
        elif path.exists():
            if path.is_file():
                path.unlink()
                print(f"✓ Removed: {path} ({description})")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"✓ Removed: {path}/ ({description})")
            removed_count += 1
    
    if removed_count == 0:
        print("✓ Already clean - nothing to remove")
    else:
        print()
        print(f"✓ Cleaned {removed_count} items")
    
    # Clean Docker images if requested
    if clean_docker:
        print()
        print("Cleaning Docker images...")
        try:
            # Check if vtk-visual-test image exists
            result = subprocess.run(
                ['docker', 'images', '-q', 'vtk-visual-test'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Remove the image
                subprocess.run(['docker', 'rmi', 'vtk-visual-test'], check=True)
                print("✓ Removed Docker image: vtk-visual-test")
                removed_count += 1
            else:
                print("✓ Docker image not found (already clean)")
        except subprocess.TimeoutExpired:
            print("⚠️  Docker timeout - skipping Docker cleanup")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to remove Docker image: {e}")
        except FileNotFoundError:
            print("⚠️  Docker not available - skipping Docker cleanup")
    
    print()
    print("=" * 80)
    print("Clean complete! To rebuild:")
    print("  1. ./setup.sh")
    print("  2. python build.py")
    if clean_docker:
        print("  3. python build.py --build-docker  # To rebuild Docker image")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clean all generated files and start fresh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py                  # Clean generated files (interactive)
  python clean.py --yes            # Clean without confirmation
  python clean.py --docker         # Also remove Docker images
  python clean.py --yes --docker   # Clean everything without confirmation
        """
    )
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--docker', action='store_true',
                       help='Also remove Docker images (vtk-visual-test)')
    
    args = parser.parse_args()
    
    try:
        # Confirmation prompt
        if not args.yes:
            msg = "\n⚠️  This will delete all generated files"
            if args.docker:
                msg += " and Docker images"
            msg += ". Continue? (yes/no): "
            response = input(msg)
            if response.lower() not in ['yes', 'y']:
                print("Cancelled.")
                sys.exit(0)
        
        clean(clean_docker=args.docker)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
