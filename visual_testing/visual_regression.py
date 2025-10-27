#!/usr/bin/env python3
"""
Visual Regression Testing for VTK Code

Compares rendered output against baseline images using:
- SSIM (Structural Similarity Index)
- Pixel difference metrics
- Automatic diff generation
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import logging

logger = logging.getLogger(__name__)


class VisualRegressionTester:
    """
    Compare VTK rendered output against baseline images
    
    Features:
    - SSIM comparison (structural similarity)
    - Pixel difference metrics
    - Diff image generation
    - Baseline management
    """
    
    def __init__(
        self,
        baseline_dir: Path,
        threshold: float = 0.95  # SSIM threshold for "pass"
    ):
        """
        Initialize visual regression tester
        
        Args:
            baseline_dir: Directory containing baseline images
            threshold: SSIM threshold (0-1, higher = more similar required)
        """
        self.baseline_dir = Path(baseline_dir)
        self.threshold = threshold
        
        # Create baseline directory if it doesn't exist
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visual regression tester initialized (threshold={threshold})")
    
    def compare_with_baseline(
        self,
        test_image: bytes,
        test_name: str,
        save_diff: bool = True
    ) -> Dict:
        """
        Compare test image with baseline
        
        Args:
            test_image: Test image as bytes (PNG)
            test_name: Name of the test
            save_diff: Whether to save diff image
            
        Returns:
            Dict with:
                - has_baseline: bool
                - similarity: float (SSIM score 0-1)
                - passed: bool (similarity >= threshold)
                - pixel_diff_count: int
                - pixel_diff_percent: float
                - diff_image_path: Optional[Path]
        """
        baseline_path = self.baseline_dir / f"{test_name}.png"
        
        # Check if baseline exists
        if not baseline_path.exists():
            logger.warning(f"No baseline found for '{test_name}'")
            return {
                'has_baseline': False,
                'similarity': 0.0,
                'passed': False,
                'pixel_diff_count': 0,
                'pixel_diff_percent': 0.0,
                'diff_image_path': None,
                'message': 'No baseline - run with update_baseline=True to create'
            }
        
        # Load images
        test_img = self._load_image_from_bytes(test_image)
        baseline_img = self._load_image(baseline_path)
        
        # Ensure same size
        if test_img.shape != baseline_img.shape:
            logger.warning(f"Image size mismatch for '{test_name}': "
                         f"test={test_img.shape}, baseline={baseline_img.shape}")
            return {
                'has_baseline': True,
                'similarity': 0.0,
                'passed': False,
                'pixel_diff_count': 0,
                'pixel_diff_percent': 0.0,
                'diff_image_path': None,
                'message': f'Size mismatch: test={test_img.shape}, baseline={baseline_img.shape}'
            }
        
        # Calculate SSIM
        similarity = self._calculate_ssim(test_img, baseline_img)
        
        # Calculate pixel differences
        pixel_diff_count, pixel_diff_percent = self._calculate_pixel_diff(
            test_img, baseline_img
        )
        
        # Generate diff image
        diff_image_path = None
        if save_diff and similarity < 1.0:
            diff_image_path = self._generate_diff_image(
                test_img, baseline_img, test_name
            )
        
        # Determine if test passed
        passed = similarity >= self.threshold
        
        result = {
            'has_baseline': True,
            'similarity': similarity,
            'passed': passed,
            'pixel_diff_count': pixel_diff_count,
            'pixel_diff_percent': pixel_diff_percent,
            'diff_image_path': diff_image_path
        }
        
        if passed:
            result['message'] = f'✓ Visual test passed (similarity={similarity:.3f})'
        else:
            result['message'] = f'✗ Visual regression detected (similarity={similarity:.3f} < {self.threshold})'
        
        return result
    
    def update_baseline(self, test_image: bytes, test_name: str) -> Path:
        """
        Update or create baseline image
        
        Args:
            test_image: Image bytes (PNG)
            test_name: Name of the test
            
        Returns:
            Path to saved baseline
        """
        baseline_path = self.baseline_dir / f"{test_name}.png"
        baseline_path.write_bytes(test_image)
        logger.info(f"Baseline updated: {baseline_path}")
        return baseline_path
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image file to numpy array"""
        img = Image.open(path)
        return np.array(img)
    
    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes to numpy array"""
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        return np.array(img)
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Returns:
            SSIM score (0-1, 1 = identical)
        """
        # Convert to grayscale if RGB
        if len(img1.shape) == 3:
            from skimage.color import rgb2gray
            img1_gray = rgb2gray(img1)
            img2_gray = rgb2gray(img2)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate SSIM
        score = ssim(img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min())
        return float(score)
    
    def _calculate_pixel_diff(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[int, float]:
        """
        Calculate pixel-level differences
        
        Returns:
            (diff_count, diff_percent)
        """
        # Calculate absolute difference
        diff = np.abs(img1.astype(float) - img2.astype(float))
        
        # Count different pixels (threshold = 5 to ignore minor differences)
        diff_pixels = np.sum(np.any(diff > 5, axis=-1) if len(diff.shape) == 3 else diff > 5)
        
        # Calculate percentage
        total_pixels = img1.shape[0] * img1.shape[1]
        diff_percent = (diff_pixels / total_pixels) * 100
        
        return int(diff_pixels), float(diff_percent)
    
    def _generate_diff_image(
        self,
        test_img: np.ndarray,
        baseline_img: np.ndarray,
        test_name: str
    ) -> Path:
        """
        Generate visual diff image showing differences
        
        Args:
            test_img: Test image array
            baseline_img: Baseline image array
            test_name: Name for output file
            
        Returns:
            Path to diff image
        """
        # Calculate absolute difference
        diff = np.abs(test_img.astype(float) - baseline_img.astype(float))
        
        # Enhance differences for visibility
        diff_enhanced = np.clip(diff * 5, 0, 255).astype(np.uint8)
        
        # Create side-by-side comparison
        if len(test_img.shape) == 3:
            # RGB images
            h, w, c = test_img.shape
            comparison = np.zeros((h, w * 3, c), dtype=np.uint8)
            comparison[:, :w] = baseline_img
            comparison[:, w:2*w] = test_img
            comparison[:, 2*w:] = diff_enhanced
        else:
            # Grayscale
            h, w = test_img.shape
            comparison = np.zeros((h, w * 3), dtype=np.uint8)
            comparison[:, :w] = baseline_img
            comparison[:, w:2*w] = test_img
            comparison[:, 2*w:] = diff_enhanced
        
        # Save diff image
        diff_path = self.baseline_dir / f"{test_name}_diff.png"
        Image.fromarray(comparison).save(diff_path)
        
        logger.info(f"Diff image saved: {diff_path}")
        return diff_path
    
    def list_baselines(self) -> list[str]:
        """List all baseline test names"""
        return [
            p.stem for p in self.baseline_dir.glob("*.png")
            if not p.stem.endswith('_diff')
        ]
    
    def baseline_exists(self, test_name: str) -> bool:
        """Check if baseline exists for test"""
        return (self.baseline_dir / f"{test_name}.png").exists()
    
    def delete_baseline(self, test_name: str) -> bool:
        """Delete baseline and diff for test"""
        baseline_path = self.baseline_dir / f"{test_name}.png"
        diff_path = self.baseline_dir / f"{test_name}_diff.png"
        
        deleted = False
        if baseline_path.exists():
            baseline_path.unlink()
            deleted = True
        if diff_path.exists():
            diff_path.unlink()
        
        return deleted
