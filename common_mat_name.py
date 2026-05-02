import logging, os 
import scipy.io as sio
from scipy.io import loadmat, savemat

def process(mat_file_path: str, expected_mat_name): 
    contents = sio.loadmat(mat_file_path)

    content_keys = list(contents.keys())
    if expected_mat_name not in content_keys:
        logging.warning(f"Variable '{expected_mat_name}' not found in {expected_mat_name}. Available keys: {list(content_keys)}. Recreating dictionary with {expected_mat_name}")
        contents[expected_mat_name] = contents.pop(content_keys[-1])
    
    print(contents.keys(), mat_file_path)
    
    savemat(mat_file_path, contents)


def common_mat_name(dataset_name, expected_mat_name="data"):
    database_path = os.path.join("database", f"{dataset_name}")
    processed_path = os.path.join("processed", f"{dataset_name}")

    files = [os.path.join(database_path, f) for f in os.listdir(database_path)]
    files.extend([os.path.join(processed_path, f) for f in os.listdir(processed_path)])

    for file in files: 
        if file.endswith(".mat"):
            process(file, expected_mat_name)

common_mat_name("ksc512_processed_34bands")