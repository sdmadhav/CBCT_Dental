"""
Preprocessing Module for CBCT Dental Images
Comprehensive image preprocessing pipeline with various techniques
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure, filters
from skimage.transform import resize, rotate

logger = logging.getLogger(__name__)


class CBCTPreprocessor:
    """
    Comprehensive preprocessing pipeline for CBCT dental images.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config or self.get_default_config()
        logger.info("CBCTPreprocessor initialized")
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default preprocessing configuration."""
        return {
            'normalization': {
                'method': 'min_max',  # 'min_max', 'z_score', 'window'
                'clip_percentile': (1, 99)
            },
            'denoising': {
                'method': 'gaussian',  # 'gaussian', 'bilateral', 'median', 'none'
                'sigma': 1.0,
                'd': 9,
                'sigma_color': 75,
                'sigma_space': 75
            },
            'contrast': {
                'method': 'clahe',  # 'clahe', 'histogram_eq', 'none'
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'windowing': {
                'bone_window': (-400, 1800),
                'soft_tissue_window': (-160, 240)
            },
            'edge_enhancement': {
                'enable': False,
                'method': 'sobel'  # 'sobel', 'canny'
            }
        }
    
    def normalize(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        clip_percentile: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image array
            method: Normalization method ('min_max', 'z_score', 'window')
            clip_percentile: Percentile for clipping outliers
            
        Returns:
            Normalized image array
        """
        method = method or self.config['normalization']['method']
        clip_percentile = clip_percentile or self.config['normalization']['clip_percentile']
        
        # Clip outliers
        if clip_percentile:
            lower, upper = np.percentile(image, clip_percentile)
            image = np.clip(image, lower, upper)
        
        if method == 'min_max':
            # Min-max normalization to [0, 1]
            img_min, img_max = image.min(), image.max()
            if img_max - img_min > 0:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = image
        
        elif method == 'z_score':
            # Z-score normalization
            mean, std = image.mean(), image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
        
        elif method == 'window':
            # Windowing (for medical images)
            window_center = (image.max() + image.min()) / 2
            window_width = image.max() - image.min()
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            normalized = np.clip(image, window_min, window_max)
            normalized = (normalized - window_min) / (window_max - window_min)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.debug(f"Normalized image using {method}")
        return normalized.astype(np.float32)
    
    def denoise(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        is_3d: bool = False
    ) -> np.ndarray:
        """
        Apply denoising filters.
        
        Args:
            image: Input image array
            method: Denoising method ('gaussian', 'bilateral', 'median', 'none')
            is_3d: Whether the image is 3D
            
        Returns:
            Denoised image array
        """
        method = method or self.config['denoising']['method']
        
        if method == 'none':
            return image
        
        if method == 'gaussian':
            sigma = self.config['denoising']['sigma']
            if is_3d:
                denoised = gaussian_filter(image, sigma=sigma)
            else:
                denoised = cv2.GaussianBlur(
                    image.astype(np.float32),
                    (0, 0),
                    sigma
                )
        
        elif method == 'bilateral':
            if is_3d:
                # Apply bilateral filter slice by slice for 3D
                denoised = np.zeros_like(image)
                for i in range(image.shape[0]):
                    denoised[i] = cv2.bilateralFilter(
                        image[i].astype(np.float32),
                        self.config['denoising']['d'],
                        self.config['denoising']['sigma_color'],
                        self.config['denoising']['sigma_space']
                    )
            else:
                denoised = cv2.bilateralFilter(
                    image.astype(np.float32),
                    self.config['denoising']['d'],
                    self.config['denoising']['sigma_color'],
                    self.config['denoising']['sigma_space']
                )
        
        elif method == 'median':
            if is_3d:
                denoised = median_filter(image, size=3)
            else:
                denoised = cv2.medianBlur(image.astype(np.float32), 5)
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        logger.debug(f"Denoised image using {method}")
        return denoised
    
    def enhance_contrast(
        self,
        image: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image array (should be normalized to [0, 1] or [0, 255])
            method: Contrast enhancement method ('clahe', 'histogram_eq', 'none')
            
        Returns:
            Contrast-enhanced image array
        """
        method = method or self.config['contrast']['method']
        
        if method == 'none':
            return image
        
        # Convert to uint8 for OpenCV methods
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(
                clipLimit=self.config['contrast']['clip_limit'],
                tileGridSize=self.config['contrast']['tile_grid_size']
            )
            enhanced = clahe.apply(image_uint8)
        
        elif method == 'histogram_eq':
            enhanced = cv2.equalizeHist(image_uint8)
        
        else:
            raise ValueError(f"Unknown contrast method: {method}")
        
        # Convert back to float
        enhanced = enhanced.astype(np.float32) / 255.0
        
        logger.debug(f"Enhanced contrast using {method}")
        return enhanced
    
    def apply_windowing(
        self,
        image: np.ndarray,
        window_type: str = 'bone'
    ) -> np.ndarray:
        """
        Apply windowing for specific tissue types.
        
        Args:
            image: Input image array (in Hounsfield Units)
            window_type: Type of window ('bone', 'soft_tissue')
            
        Returns:
            Windowed image array
        """
        if window_type == 'bone':
            window_min, window_max = self.config['windowing']['bone_window']
        elif window_type == 'soft_tissue':
            window_min, window_max = self.config['windowing']['soft_tissue_window']
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        windowed = np.clip(image, window_min, window_max)
        windowed = (windowed - window_min) / (window_max - window_min)
        
        logger.debug(f"Applied {window_type} windowing")
        return windowed
    
    def detect_edges(
        self,
        image: np.ndarray,
        method: str = 'sobel'
    ) -> np.ndarray:
        """
        Detect edges in the image.
        
        Args:
            image: Input image array
            method: Edge detection method ('sobel', 'canny')
            
        Returns:
            Edge map
        """
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        if method == 'sobel':
            sobelx = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = edges / edges.max()
        
        elif method == 'canny':
            edges = cv2.Canny(image_uint8, 100, 200)
            edges = edges.astype(np.float32) / 255.0
        
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        logger.debug(f"Detected edges using {method}")
        return edges
    
    def extract_slices(
        self,
        volume: np.ndarray,
        view: str = 'axial',
        indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Extract 2D slices from 3D volume.
        
        Args:
            volume: 3D volume array (Z, Y, X) or (Z, H, W)
            view: View type ('axial', 'coronal', 'sagittal')
            indices: Specific slice indices to extract. If None, extract all
            
        Returns:
            List of 2D slices
        """
        if view == 'axial':
            # Z-axis slices (default DICOM orientation)
            axis = 0
            max_idx = volume.shape[0]
        elif view == 'coronal':
            # Y-axis slices
            axis = 1
            max_idx = volume.shape[1]
        elif view == 'sagittal':
            # X-axis slices
            axis = 2
            max_idx = volume.shape[2]
        else:
            raise ValueError(f"Unknown view: {view}")
        
        if indices is None:
            indices = range(max_idx)
        
        slices = []
        for idx in indices:
            if idx >= max_idx:
                logger.warning(f"Slice index {idx} out of range for {view} view")
                continue
            
            if axis == 0:
                slice_2d = volume[idx, :, :]
            elif axis == 1:
                slice_2d = volume[:, idx, :]
            else:
                slice_2d = volume[:, :, idx]
            
            slices.append(slice_2d)
        
        logger.debug(f"Extracted {len(slices)} {view} slices")
        return slices
    
    def preprocess_pipeline(
        self,
        image: np.ndarray,
        is_3d: bool = False,
        custom_config: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input image array
            is_3d: Whether the image is 3D
            custom_config: Custom configuration to override defaults
            
        Returns:
            Preprocessed image array
        """
        if custom_config:
            # Temporarily update config
            original_config = self.config.copy()
            self.config.update(custom_config)
        
        try:
            # Step 1: Normalize
            processed = self.normalize(image)
            
            # Step 2: Denoise
            processed = self.denoise(processed, is_3d=is_3d)
            
            # Step 3: Enhance contrast (only for 2D)
            if not is_3d and self.config['contrast']['method'] != 'none':
                processed = self.enhance_contrast(processed)
            
            # Step 4: Edge enhancement (optional)
            if self.config['edge_enhancement']['enable'] and not is_3d:
                edges = self.detect_edges(processed, self.config['edge_enhancement']['method'])
                # Combine with original
                processed = processed + 0.3 * edges
                processed = np.clip(processed, 0, 1)
            
            logger.debug("Preprocessing pipeline completed")
            return processed
        
        finally:
            if custom_config:
                # Restore original config
                self.config = original_config


class DataAugmenter:
    """
    Data augmentation for dental images.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmenter.
        
        Args:
            config: Dictionary containing augmentation parameters
        """
        self.config = config or self.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default augmentation configuration."""
        return {
            'rotation': {'enabled': True, 'range': (-15, 15)},
            'flip': {'enabled': True, 'horizontal': True, 'vertical': False},
            'zoom': {'enabled': True, 'range': (0.9, 1.1)},
            'shift': {'enabled': True, 'range': (-0.1, 0.1)},
            'elastic': {'enabled': False, 'alpha': 100, 'sigma': 10},
            'noise': {'enabled': False, 'std': 0.01}
        }
    
    def rotate_image(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Rotate image by random angle."""
        if angle is None:
            angle = np.random.uniform(*self.config['rotation']['range'])
        return rotate(image, angle, mode='reflect', preserve_range=True)
    
    def flip_image(self, image: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Flip image along specified axis."""
        if axis is None:
            if self.config['flip']['horizontal'] and np.random.rand() > 0.5:
                image = np.fliplr(image)
            if self.config['flip']['vertical'] and np.random.rand() > 0.5:
                image = np.flipud(image)
        else:
            image = np.flip(image, axis=axis)
        return image
    
    def zoom_image(self, image: np.ndarray, zoom_factor: Optional[float] = None) -> np.ndarray:
        """Zoom image."""
        if zoom_factor is None:
            zoom_factor = np.random.uniform(*self.config['zoom']['range'])
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        resized = resize(image, (new_h, new_w), preserve_range=True)
        
        if zoom_factor > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = np.pad(resized, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
            return padded
    
    def add_noise(self, image: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise."""
        if std is None:
            std = self.config['noise']['std']
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation pipeline."""
        augmented = image.copy()
        
        if self.config['rotation']['enabled']:
            augmented = self.rotate_image(augmented)
        
        if self.config['flip']['enabled']:
            augmented = self.flip_image(augmented)
        
        if self.config['zoom']['enabled']:
            augmented = self.zoom_image(augmented)
        
        if self.config['noise']['enabled']:
            augmented = self.add_noise(augmented)
        
        return augmented


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 4096, (512, 512), dtype=np.int16)
    
    # Initialize preprocessor
    preprocessor = CBCTPreprocessor()
    
    # Preprocess
    processed = preprocessor.preprocess_pipeline(dummy_image, is_3d=False)
    print(f"Processed shape: {processed.shape}, dtype: {processed.dtype}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Initialize augmenter
    augmenter = DataAugmenter()
    
    # Augment
    augmented = augmenter.augment(processed)
    print(f"Augmented shape: {augmented.shape}")
