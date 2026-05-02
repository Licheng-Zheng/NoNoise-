import numpy as np
import sys
import os

filepath = 'indian_pine_array.npy'  
angle = 90
new_filepath = filepath.replace('.npy', f'_rotated_{angle}.npy')

def rotate_all(input_folder, output_folder, angle):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.npy', f'_rotated_{angle}.npy'))
            rotate_datacube(input_path, angle, output_path)

def rotate_datacube(filepath, angle, newfilepath):
    """ Angle must be a multiple of 90"""
    data = np.load(filepath)
    
    if data.ndim != 3:
        print(f"Data in {filepath} is not 3-dimensional has shape {data.shape}")
        return; 
    
    k = angle // 90  
    rotated_data = np.rot90(data, k=k, axes=(0, 1))
    
    # Save rotated datacube to a new file
    np.save(newfilepath, rotated_data)
    # print(f"Rotated data saved to: {newfilepath}")
# rotate_datacube(filepath, angle, new_filepath)

if __name__ == "__main__":
    rotate_all('pines_crop_test', 'pines_crop_test_rotated', 90)