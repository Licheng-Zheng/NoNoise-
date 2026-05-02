import os
import glob
import numpy as np
import scipy.io as sio
from typing import Callable, Dict, List

# Import your actual SSIM implementation
from ssim import ssim_hsi 


def load_hsi(path: str, possible_var_names: List[str]) -> np.ndarray:
    """
    Loads a .mat file and searches for the specific variable name.
    Returns a numpy array of shape (H, W, Bands).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    mat_data = sio.loadmat(path)
    
    for var in possible_var_names:
        if var in mat_data:
            data = mat_data[var]
            # Ensure float64 for precise metric calculation
            return data.astype(np.float64)
            
    raise KeyError(f"None of {possible_var_names} found in {path}. Keys found: {list(mat_data.keys())}")
