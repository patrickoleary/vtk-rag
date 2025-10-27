#!/usr/bin/env python3
"""
Fetch VTK data files from external sources.

For VTK test data: Uses the ExternalData SHA512 system
For VTK example data: Direct download from GitLab
"""

import hashlib
import json
import re
import requests
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VTKDataFetcher:
    """Fetches VTK test and example data files."""
    
    # Base URLs
    VTK_EXTERNAL_DATA_URL = "https://www.vtk.org/files/ExternalData/SHA512"
    VTK_GITHUB_RAW = "https://raw.githubusercontent.com/Kitware/VTK/master"
    VTK_EXAMPLES_GITLAB = "https://gitlab.kitware.com/vtk/vtk-examples/-/raw/master/src/Testing/Data"
    
    def __init__(self, output_dir: Path = Path("downloaded_data")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_sha512_from_github(self, data_file_path: str) -> Optional[str]:
        """
        Fetch SHA512 hash from VTK GitHub repository.
        
        Args:
            data_file_path: Path like "Data/beach.jpg"
            
        Returns:
            SHA512 hash string or None if not found
        """
        sha512_url = f"{self.VTK_GITHUB_RAW}/Testing/{data_file_path}.sha512"
        
        try:
            response = requests.get(sha512_url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
            else:
                print(f"  ⚠️  SHA512 file not found: {sha512_url}")
                return None
        except Exception as e:
            print(f"  ⚠️  Error fetching SHA512: {e}")
            return None
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from URL to output path.
        
        Args:
            url: Source URL
            output_path: Destination path
            
        Returns:
            True if successful
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            return False
    
    def fetch_test_data(self, data_file_path: str) -> Optional[Path]:
        """
        Fetch VTK test data using the ExternalData SHA512 system.
        
        Args:
            data_file_path: Path like "Data/beach.jpg"
            
        Returns:
            Path to downloaded file or None if failed
        """
        print(f"Fetching test data: {data_file_path}")
        
        # Get the SHA512 hash
        sha512 = self.fetch_sha512_from_github(data_file_path)
        if not sha512:
            return None
        
        # Download from ExternalData
        filename = Path(data_file_path).name
        output_path = self.output_dir / "test_data" / data_file_path
        
        if output_path.exists():
            print(f"  ✓ Already exists: {output_path}")
            return output_path
        
        download_url = f"{self.VTK_EXTERNAL_DATA_URL}/{sha512}"
        print(f"  📥 Downloading from: {download_url}")
        
        if self.download_file(download_url, output_path):
            print(f"  ✓ Saved to: {output_path}")
            return output_path
        
        return None
    
    def fetch_example_data(self, filename: str) -> Optional[Path]:
        """
        Fetch VTK example data from GitLab.
        
        Args:
            filename: Filename like "mug.e"
            
        Returns:
            Path to downloaded file or None if failed
        """
        print(f"Fetching example data: {filename}")
        
        output_path = self.output_dir / "example_data" / filename
        
        if output_path.exists():
            print(f"  ✓ Already exists: {output_path}")
            return output_path
        
        download_url = f"{self.VTK_EXAMPLES_GITLAB}/{filename}"
        print(f"  📥 Downloading from: {download_url}")
        
        if self.download_file(download_url, output_path):
            print(f"  ✓ Saved to: {output_path}")
            return output_path
        
        return None
    
    def get_test_data_url(self, data_file_path: str) -> Optional[str]:
        """Get the download URL for a test data file."""
        sha512 = self.fetch_sha512_from_github(data_file_path)
        if sha512:
            return f"{self.VTK_EXTERNAL_DATA_URL}/{sha512}"
        return None
    
    def get_example_data_url(self, filename: str) -> str:
        """Get the download URL for an example data file."""
        return f"{self.VTK_EXAMPLES_GITLAB}/{filename}"


def extract_data_files_from_code(code: str) -> List[str]:
    """
    Extract data file references from code, including argparse hints.
    
    Args:
        code: Python source code
        
    Returns:
        List of potential data filenames
    """
    data_files = []
    
    # Pattern 1: argparse help text with example filenames
    # e.g., help='A required filename e.g mug.e.'
    argparse_pattern = r"help\s*=\s*['\"].*?e\.g\.?\s+([a-zA-Z0-9_\-\.]+)"
    for match in re.finditer(argparse_pattern, code, re.IGNORECASE):
        filename = match.group(1).strip('.,;:')
        data_files.append(filename)
    
    # Pattern 2: SetFileName calls with Data/ prefix
    # e.g., SetFileName(VTK_DATA_ROOT + "/Data/beach.jpg")
    data_root_pattern = r'SetFileName\([^)]*["\']([^"\']+)["\']'
    for match in re.finditer(data_root_pattern, code):
        filepath = match.group(1)
        if 'Data/' in filepath:
            # Extract just the Data/filename part
            parts = filepath.split('Data/')
            if len(parts) > 1:
                data_files.append('Data/' + parts[1].strip('/'))
    
    # Pattern 3: Direct filename references in argparse
    # e.g., default='filename.ext'
    default_pattern = r"default\s*=\s*['\"]([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)['\"]"
    for match in re.finditer(default_pattern, code):
        filename = match.group(1)
        # Only include if it looks like a data file (has extension)
        if '.' in filename and not filename.endswith('.py'):
            data_files.append(filename)
    
    return list(set(data_files))  # Remove duplicates


def main():
    """Example usage of the VTKDataFetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch VTK data files")
    parser.add_argument('--test-file', help='Test data file path (e.g., Data/beach.jpg)')
    parser.add_argument('--example-file', help='Example data filename (e.g., mug.e)')
    parser.add_argument('--output-dir', default='downloaded_data', help='Output directory')
    
    args = parser.parse_args()
    
    fetcher = VTKDataFetcher(Path(args.output_dir))
    
    if args.test_file:
        fetcher.fetch_test_data(args.test_file)
    
    if args.example_file:
        fetcher.fetch_example_data(args.example_file)


if __name__ == '__main__':
    main()
