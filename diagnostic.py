import scipy.io as sio
import numpy as np 
from pathlib import Path
import warnings
# Load your two images

paths = [(r'C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\database\ksc512_processed_34bands\clean.mat', ['data']), 
         (r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\database\ksc512_processed_34bands\noisy.mat", ['data']),
         (r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\processed\ksc512_34bands\sert.mat", ['denoised_ksc'])]

def run_diagnostic_on_path(paths:list):
    for path, keys in paths:
        mat_data = sio.loadmat(path)
        for key in keys:
            try:
                data = mat_data[key]
                print(f"Loaded from {Path(path).name} key '{key}':")
                print(f"  shape={data.shape}")
                print(f"  dtype={data.dtype}")
                print(f"  min={data.min()}")
                print(f"  max={data.max()}")
                print("") # Adds a blank line (or use "" for two)
                
            except KeyError:
                print(f"Key '{key}' not found in {Path(path).name}. Available keys: {list(mat_data.keys())}")

def crawl(root_dir=".", output_filename="crawl_diagnostics.md"):
    """
    Crawls 'database' and 'processed' directories relative to root_dir,
    gathers diagnostics on all .mat files, and saves the report to a
    Markdown file.
    """
    
    root_path = Path(root_dir)
    target_dirs = [root_path / "database", root_path / "processed"]
    
    # Use 'w' mode to create or overwrite the file each time
    with open(output_filename, 'w', encoding='utf-8') as f:
        print(f"Starting crawl... Saving report to {output_filename}")
        f.write("# Data Diagnostics Crawl \n")
        
        for dir_path in target_dirs:
            if not dir_path.exists():
                print(f"Warning: Directory not found, skipping: {dir_path}")
                continue

            # Write the main H2 header for "database" or "processed"
            f.write(f"## Directory: {dir_path.name}\n")
            print(f"--- Processing {dir_path.name} ---")

            # Use rglob to find all .mat files recursively
            mat_files = sorted(list(dir_path.rglob('*.mat')))
            
            if not mat_files:
                f.write("*No .mat files found in this directory.*")
                continue

            current_subheader = None

            for file_path in mat_files:
                try:
                    # Get the parent directory name as the subheader
                    # (e.g., "ksc512", "cuprite512", "ksc512_34bands")
                    dataset_name = file_path.parent.name
                    
                    # Write the H3 subheader if it's a new dataset
                    if dataset_name != current_subheader:
                        f.write(f"### Dataset: {dataset_name}\n")
                        current_subheader = dataset_name
                        
                    # Write the specific file name
                    f.write(f"#### File: `{file_path.name}`\n")
                    
                    # Load the .mat file
                    # Suppress warnings about older MAT file formats
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mat_data = sio.loadmat(file_path)

                    f.write("```") # Start a code block for readability
                    
                    data_keys_found = 0
                    
                    # Iterate over all keys in the loaded file
                    for key, data in mat_data.items():
                        # Skip metadata keys
                        if key.startswith('__'):
                            continue
                        
                        # Only analyze actual data arrays
                        if isinstance(data, np.ndarray):
                            data_keys_found += 1
                            f.write(f"Key: '{key}'")
                            f.write(f"  shape={data.shape}")
                            f.write(f"  dtype={data.dtype}")
                            
                            # Add try-except for min/max, as it can fail
                            # on empty or non-numeric arrays
                            try:
                                f.write(f"  min={np.min(data)}")
                                f.write(f"  max={np.max(data)}")
                            except Exception as e:
                                f.write(f"  min=N/A (Error: {e})")
                                f.write(f"  max=N/A (Error: {e})")
                            
                            f.write("") # Add a newline between keys
                    
                    if data_keys_found == 0:
                        f.write("No valid data arrays found in this file.")
                    
                    f.write("```") # End the code block

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    f.write(f"**Error processing {file_path.name}**: {e}")

    print(f"✅ Diagnostic crawl complete. Results saved to {output_filename}")


# --- Main execution ---
if __name__ == "__main__":
    
    # Run the crawl function
    crawl()
    run_diagnostic_on_path(paths)
