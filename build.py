#!/usr/bin/env python3
"""
Build script - Complete pipeline from raw data to searchable index

Runs:
1. Corpus Preparation
   - prepare_corpus.py (chunk all documents)
   - analyze_corpus.py --visualize (generate statistics and charts)
   - example_usage.py (demonstrate chunk access patterns)

2. Index Building
   - build_qdrant_index.py --all (build hybrid vector + BM25 index)
   - Test query (verify index works)

3. Docker Image (optional, with --build-docker)
   - Build visual testing Docker image

Prerequisites:
- Raw data files in data/raw/
- Qdrant running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
- Dependencies installed: pip install -r build-indexes/requirements.txt
- Docker (optional, only for --build-docker)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print()
    print(Colors.BOLD + Colors.BLUE + "=" * 80 + Colors.END)
    print(Colors.BOLD + Colors.BLUE + text + Colors.END)
    print(Colors.BOLD + Colors.BLUE + "=" * 80 + Colors.END)
    print()

def print_step(text):
    print(Colors.BOLD + Colors.GREEN + f"→ {text}" + Colors.END)

def print_warning(text):
    print(Colors.YELLOW + f"⚠️  {text}" + Colors.END)

def print_error(text):
    print(Colors.RED + f"❌ {text}" + Colors.END)

def run_command(cmd, cwd=None):
    """Run a command and stream output"""
    print(Colors.BOLD + f"Running: {' '.join(cmd)}" + Colors.END)
    print()
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    
    print()
    return result

def check_prerequisites():
    """Check if prerequisites are met"""
    print_header("Checking Prerequisites")
    
    issues = []
    
    # Check raw data files
    print_step("Checking raw data files...")
    raw_files = [
        'data/raw/vtk-python-docs.jsonl',
        'data/raw/vtk-python-examples.jsonl',
        'data/raw/vtk-python-tests.jsonl'
    ]
    
    for file in raw_files:
        if not Path(file).exists():
            issues.append(f"Missing: {file}")
            print(f"  ❌ {file}")
        else:
            print(f"  ✓ {file}")
    
    # Check if Qdrant is running
    print_step("Checking Qdrant...")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        if result == 0:
            print("  ✓ Qdrant is running on localhost:6333")
        else:
            issues.append("Qdrant not running")
            print_warning("Qdrant not running on localhost:6333")
            print("         Start with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    except Exception as e:
        issues.append(f"Could not check Qdrant: {e}")
    
    # Check visualization dependencies (optional)
    print_step("Checking visualization dependencies (optional)...")
    try:
        import matplotlib
        print("  ✓ Matplotlib installed (visualizations enabled)")
    except ImportError:
        print_warning("Matplotlib not installed (visualizations will be skipped)")
        print("         Install with: pip install -r prepare-corpus/requirements.txt")
    
    # Check indexing dependencies
    print_step("Checking indexing dependencies...")
    try:
        import sentence_transformers
        import qdrant_client
        print("  ✓ Indexing dependencies installed")
    except ImportError as e:
        issues.append("Missing indexing dependencies")
        print_warning("Indexing dependencies not installed")
        print("         Install with: pip install -r build-indexes/requirements.txt")
    
    if issues:
        print()
        print_error("Prerequisites not met:")
        for issue in issues:
            print(f"  • {issue}")
        print()
        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(1)
    
    print()
    print(Colors.GREEN + "✓ Prerequisites check complete" + Colors.END)

def stage1_prepare_corpus():
    """Stage 1: Prepare corpus"""
    print_header("STAGE 1: Prepare Corpus")
    
    # Step 1: Chunk documents
    print_step("Step 1: Chunking documents...")
    run_command([sys.executable, 'prepare-corpus/prepare_corpus.py'])
    
    # Step 2: Analyze corpus
    print_step("Step 2: Analyzing corpus and generating visualizations...")
    run_command([sys.executable, 'prepare-corpus/analyze_corpus.py', '--visualize'])
    
    # Step 3: Show example usage
    print_step("Step 3: Demonstrating chunk access patterns...")
    run_command([sys.executable, 'prepare-corpus/example_usage.py'])
    
    print()
    print(Colors.GREEN + "✓ Stage 1 complete: ~131,000 chunks ready in data/processed/" + Colors.END)

def stage2_build_index():
    """Stage 2: Build hybrid index"""
    print_header("STAGE 2: Build Hybrid Index")
    
    # Step 1: Build index
    print_step("Step 1: Building hybrid index (vector + BM25)...")
    run_command([sys.executable, 'build-indexes/build_qdrant_index.py', '--all'])
    
    # Step 2: Verify index
    print_step("Step 2: Verifying index with example queries...")
    run_command([sys.executable, 'build-indexes/example_usage.py'])
    
    print()
    print(Colors.GREEN + "✓ Stage 2 complete: Hybrid index ready at http://localhost:6333/dashboard" + Colors.END)

def stage3_build_docker():
    """Stage 3: Build Docker image for visual testing"""
    print_header("STAGE 3: Build Docker Image (Visual Testing)")
    
    # Check if Docker is available
    print_step("Checking Docker availability...")
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, timeout=5)
        if result.returncode != 0:
            print_error("Docker not available")
            return False
        print(f"  ✓ {result.stdout.decode().strip()}")
    except Exception as e:
        print_error(f"Docker not available: {e}")
        return False
    
    # Build Docker image
    print_step("Building VTK visual testing Docker image...")
    print("  This may take 10-15 minutes on first build (downloading VTK, Python deps)")
    print()
    
    dockerfile_path = Path('visual_testing/Dockerfile')
    if not dockerfile_path.exists():
        print_error(f"Dockerfile not found: {dockerfile_path}")
        return False
    
    try:
        run_command([
            'docker', 'build',
            '-t', 'vtk-visual-test',
            '-f', str(dockerfile_path),
            'visual_testing/'
        ])
    except Exception as e:
        print_error(f"Docker build failed: {e}")
        return False
    
    print()
    print(Colors.GREEN + "✓ Stage 3 complete: Docker image 'vtk-visual-test' ready" + Colors.END)
    return True

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="VTK RAG Pipeline - Full Build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                    # Build corpus and index only
  python build.py --build-docker     # Also build Docker image for visual testing
        """
    )
    parser.add_argument('--build-docker', action='store_true',
                       help='Build Docker image for visual testing (adds 10-15 min)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("VTK RAG Pipeline - Full Build")
    print("This will:")
    print("  1. Chunk all documents (~131,000 chunks)")
    print("  2. Analyze corpus and generate visualizations")
    print("  3. Build hybrid search index (vector + BM25)")
    print("  4. Test the index")
    if args.build_docker:
        print("  5. Build Docker image for visual testing")
    print()
    expected_time = "15-25 minutes" if args.build_docker else "5-10 minutes"
    print(f"Expected time: {expected_time}")
    print()
    
    try:
        # Check prerequisites
        check_prerequisites()
        
        # Stage 1: Prepare corpus
        stage1_prepare_corpus()
        
        # Stage 2: Build index
        stage2_build_index()
        
        # Stage 3: Build Docker (optional)
        if args.build_docker:
            stage3_build_docker()
        
        # Summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print_header("BUILD COMPLETE! 🎉")
        print(f"Total time: {minutes}m {seconds}s")
        print()
        print("What you got:")
        print("  ✓ ~131,000 searchable chunks in data/processed/")
        print("  ✓ Visualizations in visualizations/")
        print("  ✓ Hybrid search index (vector + BM25)")
        if args.build_docker:
            print("  ✓ Docker image 'vtk-visual-test' for code execution")
        print()
        print("Next steps:")
        print("  • Query the system: python query.py 'How do I create a cylinder?'")
        print("  • View Qdrant UI: http://localhost:6333/dashboard")
        print("  • View charts: Open visualizations/*.png")
        if args.build_docker:
            print("  • Test with visual validation: python query.py 'Create sphere' --visual-test")
        print()
        
    except KeyboardInterrupt:
        print()
        print(Colors.YELLOW + "Build cancelled by user" + Colors.END)
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
