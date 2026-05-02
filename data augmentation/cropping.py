import numpy as np
import os

def crop_and_save_chunks(input_npy, output_folder, chunk_size_y, chunk_size_x, chunk_size_z,
                         contiguous=True, offset_y=50, offset_x=50,
                         contiguous_spectral=True, offset_spectral=5):
    data = np.load(input_npy)
    print(f"Data shape {data.shape}")

    os.makedirs(output_folder, exist_ok=True)

    step_y = chunk_size_y if contiguous else offset_y
    step_x = chunk_size_x if contiguous else offset_x
    step_s = chunk_size_z if contiguous_spectral else offset_spectral

    basename = os.path.basename(input_npy)
    dataset_name = os.path.splitext(basename)[0]

    h, w, s = data.shape
    count = 0

    for y in range(0, h - chunk_size_y + 1, step_y):
        for x in range(0, w - chunk_size_x + 1, step_x):
            for band in range(0, s - chunk_size_z + 1, step_s):
                chunk = data[y:y+chunk_size_y, x:x+chunk_size_x, band:band+chunk_size_z]
                filename = f"{dataset_name}_{x}_{y}_{band}.npy"
                filepath = os.path.join(output_folder, filename)
                np.save(filepath, chunk)
                count += 1

    print(f"Saved {count} chunks to {output_folder}")


if __name__ == "__main__":
    # Parameters
    input_npy = "indian_pine_array.npy"  
    output_folder = "pines_crop_test"  
    chunk_size_y = 20
    chunk_size_x = 20 
    chunk_size_z = 40 
    contiguous = True
    offset_y = 50 
    offset_x = 50 
    contiguous_spectral = True
    offset_spectral = 5

    crop_and_save_chunks(input_npy, output_folder, chunk_size_y, chunk_size_x, chunk_size_z,
                         contiguous, offset_y, offset_x,
                         contiguous_spectral, offset_spectral)