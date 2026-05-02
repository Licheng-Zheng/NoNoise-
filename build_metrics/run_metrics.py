import os
import glob
import numpy as np
import scipy.io as sio
from typing import Callable, Dict, List
import inspect
import csv

from metrics import * 
from data_loader import load_hsi

# ADD ENTROPY

def call_metric(metric_func: Callable, clean, processed, context: Dict) -> float:
    """
    Function that calls all of the metric functions. Some functions might not accept some context parameters, 
    so instead of making the user pass in arguments for everything every time, a unified context dictionary is used, 
    and only the required parameters are passed into the function. 

    Safely call a metric with only the parameters it accepts.

    - Always pass required positional args (clean, processed).
    - Filter context keys to only those present in the metric signature. (the context are all the additional parameters)
    """

    # This line gets the "parameter signature" of the metric function (all of the required parameters) 
    sig = inspect.signature(metric_func)
    accepted = {}

    # Builds a dictionary so that the required parameters are passed into the dictionary, and then passed into the function
    for k, v in context.items():
        if k in sig.parameters:
            param = sig.parameters[k]
            if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                accepted[k] = v
    return metric_func(clean, processed, **accepted)

# --- CALL METHODS (I need to split this into a separate file once everything gets cleaned up) 
# This is just a proof of concept so that I can create a lot of different calling methods in the future (unclutter the main function)
def config_mode(mode: int, dataset_name: str = None, file_paths: Dict = None) -> Dict: 
    dictionary_to_return = {}

    # 1. by path name 
    if mode == 1: 
        dictionary_to_return["clean_path"] = os.path.join("database", dataset_name, "clean.mat")
        dictionary_to_return["processed_path"] = os.path.join("processed", dataset_name, "processed.mat")
        dictionary_to_return["noisy_path"] = os.path.join("database", dataset_name, "noisy.mat")

    # by providing the paths themselves
    elif mode == 2: 
        dictionary_to_return["clean_path"] = file_paths.get("clean_path", None)
        dictionary_to_return["processed_path"] = file_paths.get("processed_path", None)
        dictionary_to_return["noisy_path"] = file_paths.get("noisy_path", None)


def main():
    # --- A. CONFIGURATION ---
    # Should probably use absolute paths here, but I can't run anything anyways so we'll see if it works
    file_path = {
        "clean_path": "database/ksc512_processed_34bands/clean.mat",
        "processed_path": "processed/ksc512_processed_34bands/processed.mat",
        "noisy_path": "database/ksc512_processed_34bands/noisy.mat"
    }

    dataset_name = "ksc512_processed_34bands"

    mode = 1 # To create the file path package where all the data is accessible for everything to be computed
    
    # Variable names to look for inside .mat files, just put anything that you've seen in here because its a hassle to change
    # the data keys for mat files
    clean_vars = ["denoised_ksc", "cube", "clean", "data"]
    processed_vars = ["denoised_ksc", "data", "cube", "reconstructed"]

    # All the metrics that we want to run (put this into a constants configuration file when this project is done) 
    metrics_to_run = {
        "PSNR": run_psnr,
        "SSIM": run_ssim, 
        "ONE": onesies, 
        "TWO": twosies
    }

    # Optional settings available to metrics. Metrics will only receive
    # the keys they accept, via call_metric's signature filtering. Put this into a constants configuration file too
    metric_context = {
        "window_size": 11,   # used by SSIM
        "normalize": True,   # used by SSIM
        "device": None,      # used by SSIM (e.g., 'cuda' or 'cpu')
        "max_val": 1.0,      # used by PSNR
        # Add more optional items as they become available (e.g., 'mask', 'band_weights')
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
    # Dynamic header based on metrics_to_run
    header_cols = ['Model Name'] + list(metrics_to_run.keys())
    header_line = " | ".join([f"{col:<15}" for col in header_cols])
    print(header_line)
    print("-" * len(header_line))

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
                score = call_metric(metric_func, clean_hsi, processed_hsi, metric_context)
                row_results[metric_name] = score

            # 4. Print Row (optional formatting), dynamic across all metrics
            row_values = [f"{model_name:<15}"]
            for m in metrics_to_run.keys():
                val = row_results.get(m, float('nan'))
                # Print floats to 4 decimals when possible
                try:
                    row_values.append(f"{float(val):<15.4f}")
                except (TypeError, ValueError):
                    row_values.append(f"{str(val):<15}")
            print(" | ".join(row_values))
            results_summary.append(row_results)

        except Exception as e:
            print(f"{model_name:<40} | [ERROR] {e}")

    # --- D. SAVE SUMMARY TO CSV ---
    # Header: 'Model Output' + metric names (based on metrics_to_run order)
    csv_headers = ['Model Output'] + list(metrics_to_run.keys())
    csv_path = os.path.join(processed_dir, 'metrics.csv')

    # Ensure directory exists (it should, but be safe)
    os.makedirs(processed_dir, exist_ok=True)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in results_summary:
            out_row = {'Model Output': row.get('name', '')}
            for m in metrics_to_run.keys():
                out_row[m] = row.get(m, float('nan'))
            writer.writerow(out_row)

    print(f"\nSaved metrics CSV to: {csv_path}")

if __name__ == "__main__":
    main()