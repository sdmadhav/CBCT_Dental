"""
DICOM Data Loader Module for CBCT Dental Images
Handles loading, caching, and metadata extraction from DICOM files
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DICOMLoader:
    """
    Loads and processes DICOM files from CBCT dental scans.
    Supports caching for faster subsequent loads.
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        expected_shape: Tuple[int, int, int] = (401, 401, 251),
        expected_dtype: str = 'int16'
    ):
        """
        Initialize DICOM loader.
        
        Args:
            data_dir: Root directory containing DICOM files
            cache_dir: Directory to store cached processed data
            use_cache: Whether to use caching
            expected_shape: Expected 3D volume shape (X, Y, Z)
            expected_dtype: Expected data type for validation
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.use_cache = use_cache
        self.expected_shape = expected_shape
        self.expected_dtype = expected_dtype
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DICOMLoader initialized: {self.data_dir}")
    
    def load_single_dicom(self, filepath: Union[str, Path]) -> Dict:
        """
        Load a single DICOM file.
        
        Args:
            filepath: Path to DICOM file
            
        Returns:
            Dictionary containing image array and metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"DICOM file not found: {filepath}")
        
        try:
            # Try pydicom first
            dcm = pydicom.dcmread(str(filepath))
            image = dcm.pixel_array
            
            metadata = {
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'manufacturer': getattr(dcm, 'Manufacturer', 'Unknown'),
                'pixel_spacing': getattr(dcm, 'PixelSpacing', [0.2, 0.2]),
                'slice_thickness': getattr(dcm, 'SliceThickness', 0.2),
                'rows': getattr(dcm, 'Rows', image.shape[0]),
                'columns': getattr(dcm, 'Columns', image.shape[1]),
                'bits_allocated': getattr(dcm, 'BitsAllocated', 16),
                'rescale_intercept': getattr(dcm, 'RescaleIntercept', 0),
                'rescale_slope': getattr(dcm, 'RescaleSlope', 1),
            }
            
            # Apply rescale if available
            if metadata['rescale_slope'] != 1 or metadata['rescale_intercept'] != 0:
                image = image * metadata['rescale_slope'] + metadata['rescale_intercept']
            
            logger.debug(f"Loaded DICOM: {filepath.name}, Shape: {image.shape}")
            
            return {
                'image': image,
                'metadata': metadata,
                'filepath': str(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM {filepath}: {str(e)}")
            raise
    
    def load_dicom_series(self, series_dir: Union[str, Path]) -> Dict:
        """
        Load a series of DICOM files as a 3D volume using SimpleITK.
        
        Args:
            series_dir: Directory containing DICOM series
            
        Returns:
            Dictionary containing 3D volume and metadata
        """
        series_dir = Path(series_dir)
        
        if not series_dir.exists():
            raise FileNotFoundError(f"Series directory not found: {series_dir}")
        
        # Check cache first
        cache_file = self.cache_dir / f"{series_dir.name}_volume.pkl"
        if self.use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
            
            if not dicom_names:
                raise ValueError(f"No DICOM files found in {series_dir}")
            
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()
            
            # Convert to numpy array
            volume = sitk.GetArrayFromImage(image_sitk)
            
            # Get metadata
            spacing = image_sitk.GetSpacing()  # (x, y, z)
            origin = image_sitk.GetOrigin()
            direction = image_sitk.GetDirection()
            
            # Read first DICOM for additional metadata
            dcm = pydicom.dcmread(dicom_names[0])
            
            metadata = {
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'series_description': getattr(dcm, 'SeriesDescription', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'CT'),
                'manufacturer': getattr(dcm, 'Manufacturer', 'Unknown'),
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'num_slices': len(dicom_names),
                'shape': volume.shape
            }
            
            result = {
                'volume': volume,
                'metadata': metadata,
                'series_dir': str(series_dir)
            }
            
            # Validate shape
            if volume.shape != self.expected_shape:
                logger.warning(
                    f"Volume shape {volume.shape} differs from expected {self.expected_shape}"
                )
            
            # Cache the result
            if self.use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Cached volume: {cache_file.name}")
            
            logger.info(f"Loaded DICOM series: {series_dir.name}, Shape: {volume.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading DICOM series from {series_dir}: {str(e)}")
            raise
    
    def load_batch(
        self,
        patient_dirs: Optional[List[Union[str, Path]]] = None,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load multiple DICOM series in batch.
        
        Args:
            patient_dirs: List of patient directories. If None, auto-discover
            max_samples: Maximum number of samples to load
            
        Returns:
            List of dictionaries containing volumes and metadata
        """
        if patient_dirs is None:
            # Auto-discover patient directories
            patient_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        patient_dirs = [Path(d) for d in patient_dirs]
        
        if max_samples:
            patient_dirs = patient_dirs[:max_samples]
        
        results = []
        logger.info(f"Loading {len(patient_dirs)} patient scans...")
        
        for patient_dir in tqdm(patient_dirs, desc="Loading DICOM series"):
            try:
                result = self.load_dicom_series(patient_dir)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {patient_dir}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(results)} scans")
        return results
    
    def validate_dicom(self, data: Dict) -> bool:
        """
        Validate loaded DICOM data.
        
        Args:
            data: Dictionary containing image/volume data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if 'volume' in data:
                volume = data['volume']
                if volume.shape != self.expected_shape:
                    logger.warning(f"Shape mismatch: {volume.shape} vs {self.expected_shape}")
                    return False
                
                if volume.dtype != self.expected_dtype:
                    logger.warning(f"Dtype mismatch: {volume.dtype} vs {self.expected_dtype}")
                    # Convert if possible
                    data['volume'] = volume.astype(self.expected_dtype)
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    def get_patient_list(self) -> List[str]:
        """
        Get list of available patient directories.
        
        Returns:
            List of patient directory names
        """
        return [d.name for d in self.data_dir.iterdir() if d.is_dir()]
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            logger.info(f"Cleared cache: {self.cache_dir}")


def convert_dicom_to_numpy(
    dicom_path: Union[str, Path],
    output_path: Union[str, Path],
    normalize: bool = True
):
    """
    Utility function to convert DICOM to numpy array and save.
    
    Args:
        dicom_path: Path to DICOM file or series directory
        output_path: Path to save numpy array
        normalize: Whether to normalize to [0, 1]
    """
    dicom_path = Path(dicom_path)
    output_path = Path(output_path)
    
    loader = DICOMLoader(dicom_path.parent if dicom_path.is_file() else dicom_path)
    
    if dicom_path.is_file():
        data = loader.load_single_dicom(dicom_path)
        array = data['image']
    else:
        data = loader.load_dicom_series(dicom_path)
        array = data['volume']
    
    if normalize:
        array = (array - array.min()) / (array.max() - array.min())
    
    np.save(output_path, array)
    logger.info(f"Saved numpy array: {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = DICOMLoader(
        data_dir="./data/raw",
        cache_dir="./data/processed/cache",
        use_cache=True
    )
    
    # Load single series
    # result = loader.load_dicom_series("./data/raw/patient_001")
    # print(f"Volume shape: {result['volume'].shape}")
    # print(f"Metadata: {result['metadata']}")
    
    # Load batch
    # results = loader.load_batch(max_samples=5)
    # print(f"Loaded {len(results)} patient scans")
    
    # Get patient list
    patients = loader.get_patient_list()
    print(f"Found {len(patients)} patients: {patients[:5]}")
