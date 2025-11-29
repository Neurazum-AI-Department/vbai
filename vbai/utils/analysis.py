"""
Brain Structure Analysis Utilities for Vbai
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from PIL import Image

import torch

# Optional imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class BrainStructureAnalyzer:
    """
    Analyzer for extracting brain structure features from MRI images.
    
    Uses edge detection and region analysis to identify
    structural characteristics useful for classification.
    
    Args:
        edge_method: Edge detection method ('canny', 'sobel', 'laplacian')
        blur_kernel: Kernel size for Gaussian blur preprocessing
    
    Example:
        >>> analyzer = BrainStructureAnalyzer()
        >>> features = analyzer.analyze(image)
        >>> print(features['edge_density'])
    """
    
    def __init__(
        self,
        edge_method: str = 'canny',
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150
    ):
        if not HAS_CV2:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")
        
        self.edge_method = edge_method
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def analyze(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze brain structure in an MRI image.
        
        Args:
            image: Input image (numpy array, PIL Image, or tensor)
        
        Returns:
            Dictionary of structural features
        """
        # Convert to numpy grayscale
        gray = self._to_grayscale(image)
        
        # Apply blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Edge detection
        edges = self._detect_edges(blurred)
        
        # Extract features
        features = {
            'edge_density': self._compute_edge_density(edges),
            'edge_mean_intensity': self._compute_mean_intensity(edges),
            'symmetry_score': self._compute_symmetry(gray),
            'texture_variance': self._compute_texture_variance(gray),
            'brain_area_ratio': self._compute_brain_ratio(gray),
            'contrast': self._compute_contrast(gray),
            'homogeneity': self._compute_homogeneity(gray),
        }
        
        return features
    
    def get_edge_map(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """
        Get edge detection map for visualization.
        
        Args:
            image: Input image
        
        Returns:
            Edge map as numpy array
        """
        gray = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return self._detect_edges(blurred)
    
    def _to_grayscale(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            if image.dim() == 3:
                # Assume RGB, convert to grayscale
                image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert('L'))
        else:
            if len(image.shape) == 3:
                img_np = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img_np = image
        
        return img_np
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply edge detection."""
        if self.edge_method == 'canny':
            return cv2.Canny(gray, self.canny_low, self.canny_high)
        elif self.edge_method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        elif self.edge_method == 'laplacian':
            return np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge method: {self.edge_method}")
    
    def _compute_edge_density(self, edges: np.ndarray) -> float:
        """Compute ratio of edge pixels to total pixels."""
        return np.sum(edges > 0) / edges.size
    
    def _compute_mean_intensity(self, edges: np.ndarray) -> float:
        """Compute mean intensity of edge pixels."""
        return float(np.mean(edges))
    
    def _compute_symmetry(self, gray: np.ndarray) -> float:
        """Compute left-right symmetry score."""
        h, w = gray.shape
        left = gray[:, :w//2]
        right = gray[:, w//2:]
        right_flipped = np.fliplr(right)
        
        # Handle odd width
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
        
        # Compute correlation
        diff = np.abs(left.astype(float) - right_flipped.astype(float))
        symmetry = 1.0 - (np.mean(diff) / 255.0)
        
        return float(symmetry)
    
    def _compute_texture_variance(self, gray: np.ndarray) -> float:
        """Compute texture variance using local variance."""
        # Use a sliding window approach
        kernel_size = 5
        local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2
        
        return float(np.mean(local_var))
    
    def _compute_brain_ratio(self, gray: np.ndarray) -> float:
        """Estimate ratio of brain tissue to background."""
        # Simple thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return np.sum(binary > 0) / binary.size
    
    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Compute image contrast."""
        return float(np.std(gray))
    
    def _compute_homogeneity(self, gray: np.ndarray) -> float:
        """Compute texture homogeneity using GLCM-like approach."""
        # Simplified homogeneity based on neighbor differences
        diff_h = np.abs(np.diff(gray.astype(float), axis=1))
        diff_v = np.abs(np.diff(gray.astype(float), axis=0))
        
        # Homogeneity inversely related to differences
        homogeneity = 1.0 / (1.0 + np.mean(diff_h) + np.mean(diff_v))
        
        return float(homogeneity)


def extract_brain_features(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    detailed: bool = False
) -> Dict[str, float]:
    """
    Quick function to extract brain features.
    
    Args:
        image: Input MRI image
        detailed: Whether to include detailed features
    
    Returns:
        Feature dictionary
    """
    analyzer = BrainStructureAnalyzer()
    features = analyzer.analyze(image)
    
    if not detailed:
        # Return only key features
        return {
            'edge_density': features['edge_density'],
            'symmetry_score': features['symmetry_score'],
            'contrast': features['contrast'],
        }
    
    return features


def compare_brain_images(
    image1: Union[np.ndarray, Image.Image],
    image2: Union[np.ndarray, Image.Image]
) -> Dict[str, float]:
    """
    Compare two brain MRI images.
    
    Args:
        image1: First image
        image2: Second image
    
    Returns:
        Similarity metrics
    """
    analyzer = BrainStructureAnalyzer()
    
    features1 = analyzer.analyze(image1)
    features2 = analyzer.analyze(image2)
    
    # Compute differences
    comparison = {}
    for key in features1:
        comparison[f'{key}_diff'] = abs(features1[key] - features2[key])
    
    # Overall similarity
    diffs = list(comparison.values())
    comparison['overall_similarity'] = 1.0 - np.mean(diffs)
    
    return comparison
