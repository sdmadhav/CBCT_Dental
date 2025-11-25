"""
Complete CBCT Dental Image Analysis Pipeline
All modules in one file for easy deployment

Directory Structure:
cbct_pipeline/
├── src/
│   ├── data_loader.py (see previous artifact)
│   ├── preprocessing.py (see previous artifact)
│   ├── feature_extraction.py (below)
│   ├── dataset.py (below)
│   ├── models.py (below)
│   ├── train.py (below)
│   ├── evaluate.py (below)
│   ├── inference.py (below)
│   └── utils.py (below)
├── configs/
│   └── config.yaml (below)
├── main.py (below)
└── requirements.txt (below)
"""

# ============================================================================
# feature_extraction.py
# ============================================================================

"""
Feature Extraction Module for CBCT Dental Images
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract traditional and deep features from dental images."""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.cnn_extractor = None
        logger.info(f"FeatureExtractor initialized on {self.device}")
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using Gray Level Co-occurrence Matrix (GLCM).
        
        Args:
            image: Input image (2D grayscale)
            
        Returns:
            Dictionary of texture features
        """
        # Convert to uint8 if needed
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Compute GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image_uint8, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract properties
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'ASM': graycoprops(glcm, 'ASM').mean()
        }
        
        return features
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract shape descriptors from binary mask.
        
        Args:
            mask: Binary mask (2D)
            
        Returns:
            Dictionary of shape features
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'area': 0, 'perimeter': 0, 'circularity': 0, 'aspect_ratio': 0}
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio
        }
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal-to-Noise Ratio (simplified)
        mean = np.mean(image)
        std = np.std(image)
        snr = mean / std if std > 0 else 0
        
        # Blur detection using Laplacian variance
        if len(image.shape) == 2:
            laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_64F)
            blur_score = laplacian.var()
        else:
            blur_score = 0
        
        return {
            'snr': float(snr),
            'blur_score': float(blur_score),
            'mean_intensity': float(mean),
            'std_intensity': float(std)
        }
    
    def initialize_cnn_extractor(self, model_name: str = 'resnet18'):
        """Initialize CNN feature extractor."""
        if model_name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove final classification layer
            self.cnn_extractor = nn.Sequential(*list(model.children())[:-1])
        
        self.cnn_extractor = self.cnn_extractor.to(self.device)
        self.cnn_extractor.eval()
        logger.info(f"CNN extractor initialized: {model_name}")
    
    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep features using pretrained CNN.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Feature vector
        """
        if self.cnn_extractor is None:
            self.initialize_cnn_extractor()
        
        # Prepare image
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.cnn_extractor(image_tensor)
        
        return features.cpu().numpy().flatten()
    
    def detect_roi(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Detect regions of interest (teeth) using simple thresholding.
        
        Args:
            image: Input image
            threshold: Threshold for segmentation
            
        Returns:
            Binary mask and list of bounding boxes
        """
        # Normalize image
        if image.max() > 1.0:
            image_norm = (image - image.min()) / (image.max() - image.min())
        else:
            image_norm = image
        
        # Apply threshold
        mask = (image_norm > threshold).astype(np.uint8)
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        return mask, bboxes


# ============================================================================
# dataset.py
# ============================================================================

"""
PyTorch Dataset Classes for CBCT Data
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CBCTDataset2D(Dataset):
    """Dataset for 2D slice-based training."""
    
    def __init__(
        self,
        data_list: List[Dict],
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256),
        mode: str = 'train'
    ):
        """
        Initialize 2D dataset.
        
        Args:
            data_list: List of dictionaries containing image paths and labels
            transform: Optional transform to apply
            target_size: Target image size
            mode: 'train', 'val', or 'test'
        """
        self.data_list = data_list
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        
        logger.info(f"CBCTDataset2D initialized: {len(data_list)} samples ({mode})")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        
        # Load image
        if isinstance(item['image'], str):
            image = np.load(item['image'])
        else:
            image = item['image']
        
        # Resize if needed
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
        
        # Convert to tensor
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # Add channel dimension
        else:
            image = image.transpose(2, 0, 1)  # HWC to CHW
        
        image_tensor = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Get label
        label = torch.tensor(item.get('label', 0), dtype=torch.long)
        
        return image_tensor, label


class CBCTDataset3D(Dataset):
    """Dataset for 3D volume-based training."""
    
    def __init__(
        self,
        data_list: List[Dict],
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        mode: str = 'train'
    ):
        """
        Initialize 3D dataset.
        
        Args:
            data_list: List of dictionaries containing volume paths and labels
            transform: Optional transform to apply
            target_size: Target volume size
            mode: 'train', 'val', or 'test'
        """
        self.data_list = data_list
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        
        logger.info(f"CBCTDataset3D initialized: {len(data_list)} volumes ({mode})")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        
        # Load volume
        if isinstance(item['volume'], str):
            volume = np.load(item['volume'])
        else:
            volume = item['volume']
        
        # Resize if needed (simple downsampling)
        if volume.shape != self.target_size:
            zoom_factors = [t / s for t, s in zip(self.target_size, volume.shape)]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        # Add channel dimension
        volume = volume[np.newaxis, ...]
        volume_tensor = torch.from_numpy(volume).float()
        
        # Apply transforms
        if self.transform:
            volume_tensor = self.transform(volume_tensor)
        
        # Get label
        label = torch.tensor(item.get('label', 0), dtype=torch.long)
        
        return volume_tensor, label


def create_data_splits(
    data_list: List[Dict],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        data_list: List of data samples
        split_ratios: Tuple of (train, val, test) ratios
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    if shuffle:
        random.seed(seed)
        data_list = data_list.copy()
        random.shuffle(data_list)
    
    n_total = len(data_list)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    splits = {
        'train': data_list[:n_train],
        'val': data_list[n_train:n_train + n_val],
        'test': data_list[n_train + n_val:]
    }
    
    logger.info(f"Data splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def get_dataloaders(
    data_splits: Dict[str, List[Dict]],
    batch_size: int = 8,
    num_workers: int = 4,
    dataset_type: str = '2d',
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.
    
    Args:
        data_splits: Dictionary with train/val/test data lists
        batch_size: Batch size
        num_workers: Number of worker processes
        dataset_type: '2d' or '3d'
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Dictionary of DataLoaders
    """
    DatasetClass = CBCTDataset2D if dataset_type == '2d' else CBCTDataset3D
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split in data_splits and data_splits[split]:
            dataset = DatasetClass(data_splits[split], mode=split, **dataset_kwargs)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
    
    return dataloaders


# ============================================================================
# models.py
# ============================================================================

"""
Model Architectures for CBCT Analysis
"""

import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class ResNetClassifier2D(nn.Module):
    """2D CNN based on ResNet for slice classification."""
    
    def __init__(self, num_classes: int = 32, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            num_features = 512
        elif backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            num_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Modify first conv for single channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC layer
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        logger.info(f"ResNetClassifier2D initialized: {backbone}, classes={num_classes}")
    
    def forward(self, x):
        return self.backbone(x)


class UNet2D(nn.Module):
    """2D U-Net for tooth segmentation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec4 = self.conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self.conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self.conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self.conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Final conv
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
        logger.info(f"UNet2D initialized: in_channels={in_channels}, classes={num_classes}")
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([F.interpolate(b, e4.shape[2:]), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, e3.shape[2:]), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:]), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:]), e1], dim=1))
        
        return self.final(d1)


class CNN3D(nn.Module):
    """Lightweight 3D CNN for volumetric analysis."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 32):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        logger.info(f"CNN3D initialized: in_channels={in_channels}, classes={num_classes}")
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    models = {
        'resnet18_2d': ResNetClassifier2D,
        'resnet50_2d': lambda **kw: ResNetClassifier2D(backbone='resnet50', **kw),
        'unet_2d': UNet2D,
        'cnn_3d': CNN3D
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)


# Continue in next message...
