import numpy as np
from scipy.io import savemat
import os 

def convert_npy_to_mat(npy_file_path: str, mat_file_path: str, var_name: str = 'data'):
    """
    Convert a .npy file to a .mat file.

    Parameters:
        npy_file_path: Path to the input .npy file.
        mat_file_path: Path to the output .mat file.
        var_name: Variable name to use in the .mat file.
    """
    # Load the .npy file
    npy_data = np.load(npy_file_path)

    # Save it to a .mat file
    savemat(mat_file_path, {var_name: npy_data})


dataset_folder_path = r"database" 

# Currently only supports npy into mat, but I'll add more in the future as we add models that require different
# input formats 

datasets = os.listdir(dataset_folder_path)
for dataset in datasets: 
    # Find the different files in the dataset folder 
    datasets = os.listdir(os.path.join(dataset_folder_path, dataset))
    for file in datasets: 
        if file.endswith(".npy"): 
            npy_file_path = os.path.join(dataset_folder_path, dataset, file)
            mat_file_name = file.replace(".npy", ".mat")
            mat_file_path = os.path.join(dataset_folder_path, dataset, mat_file_name)
            convert_npy_to_mat(npy_file_path, mat_file_path)