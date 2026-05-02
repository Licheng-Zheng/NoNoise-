import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Renamed function from 'apply_shift' to 'apply_amplitude_shift' for clarity and removed angle parameter as we use amplitude shift
def apply_amplitude_shift(image, mean, sigma_left, sigma_right, shift_type = 1):
    shift_value = asymmetric_normal(mean, sigma_left, sigma_right)
    if shift_type == 1:
        # print("FORST")
        # print(pd.DataFrame(image[0]))
        shifted_image = image + shift_value  
        # print(f"SHIFT {shift_value}")
        # print("SECOND")
        # print(pd.DataFrame(shifted_image[0]))  
        # exit()
        return shifted_image, shift_value
    else:
        shifted_image = image * (1 + shift_value) 
        return shifted_image, shift_value

def amp_all(input_folder, output_folder, mean, sigma_left, sigma_right):
    plot_asymmetric_normal(mean, sigma_left, sigma_right)
    
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(output_folder, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))

    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            input_path = os.path.join(input_folder, filename)
            image = np.load(input_path)
            shifted_image, shift = apply_amplitude_shift(image, mean, sigma_left, sigma_right)
            output_path = os.path.join(output_folder, filename.replace('.npy', f'_ampshifted_{int(shift)}.npy')) 
            np.save(output_path, shifted_image)

def asymmetric_normal(mean, sigma_left, sigma_right, size=1):
    p_left = sigma_left / (sigma_left + sigma_right)
    u = np.random.uniform(0, 1, size)
    z = np.abs(np.random.normal(0, 1, size))

    samples = np.where(
        u < p_left,
        mean - z * sigma_left, 
        mean + z * sigma_right  
    )
    
    return samples if size > 1 else samples[0]

def plot_asymmetric_normal(mean, sigma_left, sigma_right, x_range=10, num_points=1000):
    x = np.linspace(mean - x_range, mean + x_range, num_points)
    
    A = np.sqrt(2 / np.pi) / (sigma_left + sigma_right)
    
    pdf = np.where(
        x < mean,
        A * np.exp(-0.5 * ((x - mean) / sigma_left)**2),
        A * np.exp(-0.5 * ((x - mean) / sigma_right)**2)
    )
    plt.plot(x, pdf, color='blue')
    plt.title(f"Wonky Normal Distribution (mean={mean}, sigma left={sigma_left}, sigma right={sigma_right})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    mean = 1000
    sigma_left = 5
    sigma_right = 2

    amp_all('pines_crop_test', 'pines_crop_test_ampshifted', mean, sigma_left, sigma_right)
