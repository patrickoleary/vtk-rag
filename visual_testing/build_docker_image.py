#!/usr/bin/env python3
"""
Build Docker image for visual regression testing

Usage:
    python build_docker_image.py
"""

from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from docker_sandbox import DockerSandbox


def main():
    print("=" * 80)
    print("Building VTK Visual Testing Docker Image")
    print("=" * 80)
    print()
    
    # Initialize sandbox
    sandbox = DockerSandbox(image_name="vtk-visual-testing")
    
    # Check if image already exists
    if sandbox.image_exists():
        print(f"Image 'vtk-visual-testing' already exists.")
        response = input("Rebuild? (y/N): ")
        if response.lower() != 'y':
            print("Skipping build")
            return 0
    
    # Build image
    dockerfile_dir = Path(__file__).parent
    print(f"\nBuilding from: {dockerfile_dir}")
    print("This may take 5-10 minutes...")
    print()
    
    success = sandbox.build_image(dockerfile_dir)
    
    if success:
        print()
        print("=" * 80)
        print("✅ Docker image built successfully!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Create baselines:")
        print("     UPDATE_BASELINES=1 RUN_VISUAL_TESTS=1 pytest test_visual_regression.py")
        print()
        print("  2. Run visual tests:")
        print("     RUN_VISUAL_TESTS=1 pytest test_visual_regression.py")
        return 0
    else:
        print()
        print("=" * 80)
        print("❌ Failed to build Docker image")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
