"""
Feature Engineering Module

This module provides utilities for extracting meaningful features from Chandrayaan-2 data
for ice detection and classification.
"""

import numpy as np
from typing import Dict, Tuple
import logging
from scipy import ndimage
from skimage import filters, feature, morphology

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from remote sensing data for ice detection.
    """
    
    @staticmethod
    def extract_texture_features(data: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
        """
        Extract texture features from imagery data.
        
        Args:
            data (np.ndarray): Input image data
            window_size (int): Size of the local window for feature extraction
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing various texture features
        """
        features = {}
        
        # Contrast
        features['contrast'] = filters.sobel(data)
        
        # Edges
        features['edges'] = feature.canny(data, sigma=1.0)
        
        # Local variance
        features['variance'] = ndimage.generic_filter(data, np.var, size=window_size)
        
        # Entropy
        features['entropy'] = ndimage.generic_filter(data, FeatureExtractor._entropy, size=window_size)
        
        logger.info("Texture features extracted")
        return features
    
    @staticmethod
    def _entropy(data: np.ndarray) -> float:
        """Calculate entropy of a local region."""
        hist, _ = np.histogram(data, bins=10)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def extract_radar_features(sar_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features specific to SAR radar data.
        
        Args:
            sar_data (np.ndarray): SAR imagery
            
        Returns:
            Dict[str, np.ndarray]: SAR-specific features
        """
        features = {}
        
        # Backscatter intensity
        features['intensity'] = np.abs(sar_data)
        
        # Speckle noise characteristics (multi-look processing)
        features['smoothness'] = ndimage.gaussian_filter(sar_data, sigma=3)
        
        # SAR coherence simulation
        features['coherence'] = np.abs(sar_data) / (np.abs(sar_data) + 0.1)
        
        # Phase information if available
        if np.iscomplexobj(sar_data):
            features['phase'] = np.angle(sar_data)
        
        logger.info("SAR radar features extracted")
        return features
    
    @staticmethod
    def extract_topographic_features(dem_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract topographic features from Digital Elevation Model.
        
        Args:
            dem_data (np.ndarray): DEM elevation data
            
        Returns:
            Dict[str, np.ndarray]: Topographic features
        """
        features = {}
        
        # Slope
        gy, gx = np.gradient(dem_data)
        features['slope'] = np.sqrt(gx**2 + gy**2)
        
        # Aspect (direction of slope)
        features['aspect'] = np.arctan2(gy, gx)
        
        # Curvature (second derivatives)
        gyy, gyx = np.gradient(gy)
        gxx, gxy = np.gradient(gx)
        features['curvature'] = gxx + gyy
        
        # Elevation roughness
        features['roughness'] = ndimage.generic_filter(dem_data, np.std, size=5)
        
        # Topographic Wetness Index
        features['twi'] = np.log((dem_data + 1) / (features['slope'] + 1))
        
        logger.info("Topographic features extracted")
        return features
    
    @staticmethod
    def extract_spectral_indices(optical_data: np.ndarray, band_indices: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        """
        Extract spectral indices from optical data.
        
        Args:
            optical_data (np.ndarray): Multi-band optical imagery
            band_indices (Dict[str, int]): Mapping of band names to indices
            
        Returns:
            Dict[str, np.ndarray]: Computed spectral indices
        """
        features = {}
        
        if band_indices is None:
            # Default band indices for typical sensors
            band_indices = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}
        
        # NDVI (Normalized Difference Vegetation Index) - for surface features
        if 'red' in band_indices and 'nir' in band_indices:
            red = optical_data[band_indices['red']].astype(float)
            nir = optical_data[band_indices['nir']].astype(float)
            features['ndvi'] = (nir - red) / (nir + red + 1e-8)
        
        # Normalized Difference Index
        if 'blue' in band_indices and 'red' in band_indices:
            blue = optical_data[band_indices['blue']].astype(float)
            red = optical_data[band_indices['red']].astype(float)
            features['ndi'] = (red - blue) / (red + blue + 1e-8)
        
        logger.info("Spectral indices computed")
        return features
    
    @staticmethod
    def extract_morphological_features(data: np.ndarray, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Extract morphological features using image processing.
        
        Args:
            data (np.ndarray): Input data
            threshold (float): Threshold for binary image creation
            
        Returns:
            Dict[str, np.ndarray]: Morphological features
        """
        features = {}
        
        # Binary image
        binary = data > threshold
        
        # Erosion and dilation
        features['erosion'] = morphology.erosion(binary)
        features['dilation'] = morphology.dilation(binary)
        features['opening'] = morphology.opening(binary)
        features['closing'] = morphology.closing(binary)
        
        # Skeleton
        features['skeleton'] = morphology.skeletonize(binary)
        
        logger.info("Morphological features extracted")
        return features


class MultimodalFeatureExtractor:
    """
    Extract combined features from multiple data modalities.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_extractor = FeatureExtractor()
    
    def extract_all_features(self, sar_data: np.ndarray = None,
                           optical_data: np.ndarray = None,
                           dem_data: np.ndarray = None,
                           thermal_data: np.ndarray = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all available data modalities.
        
        Args:
            sar_data (np.ndarray): SAR imagery
            optical_data (np.ndarray): Optical imagery
            dem_data (np.ndarray): DEM data
            thermal_data (np.ndarray): Thermal imagery
            
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Features for each modality
        """
        all_features = {}
        
        if sar_data is not None:
            logger.info("Extracting SAR features...")
            all_features['sar'] = self.feature_extractor.extract_radar_features(sar_data)
            all_features['sar'].update(self.feature_extractor.extract_texture_features(np.abs(sar_data)))
        
        if optical_data is not None:
            logger.info("Extracting optical features...")
            all_features['optical'] = self.feature_extractor.extract_spectral_indices(optical_data)
            all_features['optical'].update(self.feature_extractor.extract_texture_features(optical_data[0]))
        
        if dem_data is not None:
            logger.info("Extracting topographic features...")
            all_features['dem'] = self.feature_extractor.extract_topographic_features(dem_data)
        
        if thermal_data is not None:
            logger.info("Extracting thermal features...")
            all_features['thermal'] = self.feature_extractor.extract_texture_features(thermal_data[0])
        
        return all_features
    
    @staticmethod
    def stack_features(features_dict: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Stack all features into a single array for model input.
        
        Args:
            features_dict (Dict[str, Dict[str, np.ndarray]]): Features from all modalities
            
        Returns:
            np.ndarray: Stacked feature array of shape (H, W, num_features)
        """
        feature_stack = []
        
        for modality, features in features_dict.items():
            for feature_name, feature_array in features.items():
                if feature_array.ndim == 2:
                    feature_stack.append(feature_array)
                elif feature_array.ndim == 3:
                    # Handle multi-band features
                    for band in range(feature_array.shape[0]):
                        feature_stack.append(feature_array[band])
        
        stacked = np.stack(feature_stack, axis=-1)
        logger.info(f"Features stacked into shape: {stacked.shape}")
        return stacked
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """
        Normalize features across all channels.
        
        Args:
            features (np.ndarray): Stacked features
            
        Returns:
            np.ndarray: Normalized features
        """
        # Z-score normalization per feature
        normalized = np.zeros_like(features, dtype=np.float32)
        
        for i in range(features.shape[-1]):
            feature_channel = features[..., i]
            mean = np.nanmean(feature_channel)
            std = np.nanstd(feature_channel)
            
            if std > 0:
                normalized[..., i] = (feature_channel - mean) / std
            else:
                normalized[..., i] = feature_channel - mean
        
        logger.info("Features normalized")
        return normalized


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data and extract features
    sample_sar = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    sample_dem = np.random.rand(100, 100) * 1000
    
    extractor = MultimodalFeatureExtractor()
    features = extractor.extract_all_features(sar_data=sample_sar, dem_data=sample_dem)
    
    print(f"Extracted features:")
    for modality, feat_dict in features.items():
        print(f"  {modality}: {len(feat_dict)} features")
