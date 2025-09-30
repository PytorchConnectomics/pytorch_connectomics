"""
MONAI transforms for connectomics I/O operations.

Custom MONAI transforms for loading various connectomics data formats (HDF5, TIFF, etc.).
"""

from typing import Any, Dict
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from .volume import read_volume


class LoadVolumed(MapTransform):
    """
    Custom MONAI loader for connectomics volume data (HDF5, TIFF, etc.).
    
    This transform uses the connectomics read_volume function to load various
    file formats and ensures the data has a channel dimension.
    
    Args:
        keys: Keys to load from the data dictionary
        allow_missing_keys: Whether to allow missing keys in the dictionary
        
    Examples:
        >>> transform = LoadVolumed(keys=['image', 'label'])
        >>> data = {'image': 'img.h5', 'label': 'lbl.h5'}
        >>> result = transform(data)
        >>> # result['image'] shape: (C, D, H, W)
    """
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load volume data from file paths.
        
        Args:
            data: Dictionary with file paths as values
            
        Returns:
            Dictionary with loaded numpy arrays (with channel dimension)
        """
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], str):
                # Use the connectomics read_volume function
                volume = read_volume(d[key])
                # Ensure we have at least 4 dimensions (add channel if needed)
                if volume.ndim == 3:
                    volume = np.expand_dims(volume, axis=0)  # Add channel dimension
                d[key] = volume
        return d


__all__ = [
    'LoadVolumed',
]
