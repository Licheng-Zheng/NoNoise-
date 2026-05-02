from pathlib import Path
import numpy as np
from scipy.io import savemat
import os

def change_npy_file_extension_to_mat(file_path: str, save_path: str = None):
    # 1. Load the NumPy file
    data = np.load(file_path)

    # 2. Prepare the dictionary for MATLAB
    # The key 'my_variable' is how the data will be named inside MATLAB
    variable_dict = {"my_variable": data}

    # 3. Save as a .mat file
    if save_path is None:
        save_path = file_path.rsplit('.', 1)[0] + '.mat'
    
    if os.path.exists(save_path):
        print("Overwriting existing file:", save_path)
    
    savemat(save_path, variable_dict)


def change_all_in_directory(directory_path: str):
    path = Path(directory_path)
    for file in path.glob('*.npy'):
        change_npy_file_extension_to_mat(str(file))                                 