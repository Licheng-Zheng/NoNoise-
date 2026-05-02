import numpy as np
import os
import sys

# CHANGE THIS INTO A FUNCTION SO I CAN USE IT ELSEWHERE
# really basic dianostics for now but we can see what we need in the future
npy_file = 'indian_pine_array.npy'  

data = np.load(npy_file)

if data.ndim != 3:
    print(f" {npy_file} is not 3-dimensional. Shape is {data.shape}")

x, y, z = data.shape
file_size_bytes = os.path.getsize(npy_file)
file_size_mb = file_size_bytes / (1024 ** 2)

print(f"Diagnostics for {npy_file}:")
print(f"Dimensions (x, y, z): {x}, {y}, {z}")
print(f"File size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)")

