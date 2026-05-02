import os
import glob
import numpy as np
import scipy.io as sio
from typing import Callable, Dict, List

# Import your actual SSIM implementation
from Model_Evaluation.ssim import ssim_hsi 

# ==========================================
# 1. DATA LOADING HELPER
# ==========================================
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

# ==========================================
# 2. METRIC DEFINITIONS
# Define your metrics here as simple functions
# ==========================================

def run_psnr(clean: np.ndarray, processed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio."""
    mse = np.mean((clean - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0 # Adjust if your data is 0-255
    return 20 * np.log10(max_val / np.sqrt(mse))

def run_ssim(clean: np.ndarray, processed: np.ndarray) -> float:
    """Wrapper for your imported SSIM function."""
    # You can hardcode your preferred SSIM settings here
    return ssim_hsi(clean, processed, window_size=11, normalize=True)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # --- A. CONFIGURATION ---
    dataset_name = "ksc512" 
    
    # Paths
    clean_mat_path = os.path.join("database", dataset_name, "clean.mat")
    processed_dir = os.path.join("processed", dataset_name)
    
    # Variable names to look for inside .mat files
    clean_vars = ["denoised_ksc", "cube", "clean"]
    processed_vars = ["data", "cube", "reconstructed"]

    # List of metrics to run (Dictionary allows for nice printing names)
    metrics_to_run = {
        "PSNR": run_psnr,
        "SSIM": run_ssim
    }

    # --- B. LOAD GROUND TRUTH ---
    print(f"Loading Clean Reference from: {clean_mat_path}")
    try:
        clean_hsi = load_hsi(clean_mat_path, clean_vars)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load clean data. {e}")
        return

    # --- C. PROCESS ALL MODELS ---
    # Find all .mat files in the processed directory
    model_files = glob.glob(os.path.join(processed_dir, "*.mat"))
    
    if not model_files:
        print(f"No .mat files found in {processed_dir}")
        return

    print(f"\nFound {len(model_files)} models. Starting evaluation...\n")
    print(f"{'Model Name':<40} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 65)

    results_summary = []

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        
        try:
            # 1. Load Model Output
            processed_hsi = load_hsi(model_path, processed_vars)
            
            # 2. Safety Check: Shapes
            if clean_hsi.shape != processed_hsi.shape:
                print(f"{model_name:<40} | [SKIP] Shape mismatch: {clean_hsi.shape} vs {processed_hsi.shape}")
                continue

            # 3. Run Metrics
            row_results = {"name": model_name}
            
            for metric_name, metric_func in metrics_to_run.items():
                score = metric_func(clean_hsi, processed_hsi)
                row_results[metric_name] = score

            # 4. Print Row
            print(f"{model_name:<40} | {row_results['PSNR']:<10.4f} | {row_results['SSIM']:<10.4f}")
            results_summary.append(row_results)

        except Exception as e:
            print(f"{model_name:<40} | [ERROR] {e}")

    # --- D. (Optional) Next Steps ---
    # You could save 'results_summary' to a CSV here if you wanted.

if __name__ == "__main__":
    main()