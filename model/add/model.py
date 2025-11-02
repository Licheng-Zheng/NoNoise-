import os
import numpy as np
import scipy.io as sio
from typing import Optional, Tuple


def model(
    input_mat_path: str,
    input_var: str,
    output_mat_path: Optional[str] = None,
    output_var: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Load a .mat file, add 1 to every element of the specified variable, and save a new .mat file.

    Parameters:
        input_mat_path: Path to the input .mat file.
        input_var: Variable name inside the .mat file to process.
        output_mat_path: Optional path for the output .mat file. If None, appends "_plus_one.mat" to input filename.
        output_var: Optional variable name for the saved array in the output .mat. If None, uses f"{input_var}_plus_one".

    Returns:
        A tuple (output_mat_path, output_var) indicating where the new .mat file was saved and the variable name used.

    Raises:
        FileNotFoundError: If the input .mat file does not exist.
        KeyError: If the specified variable is not in the .mat file.
        Exception: For other unexpected errors during load/save operations.
    """
    if not os.path.exists(input_mat_path):
        raise FileNotFoundError(f"File not found: {input_mat_path}")

    # Derive defaults
    if output_mat_path is None:
        base, ext = os.path.splitext(input_mat_path)
        output_mat_path = f"{base}_plus_one.mat"
    if output_var is None:
        output_var = f"{input_var}_plus_one"

    # Load input .mat (squeeze_me to simplify MATLAB scalars/cells when safe)
    mat_data = sio.loadmat(input_mat_path)

    if input_var not in mat_data:
        available = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(
            f"Variable '{input_var}' not found in {input_mat_path}. Available: {available}"
        )

    arr = mat_data[input_var]

    # Core operation: add 1 elementwise
    result = np.asarray(arr) + 1

    # Save to a new .mat file
    # Use do_compression=True to reduce size; ensure it's a plain ndarray
    sio.savemat(output_mat_path, {output_var: np.asarray(result)}, do_compression=True)

    return output_mat_path, output_var


if __name__ == "__main__":
    # Simple CLI usage for manual testing:
    # Defaults match the previous behavior but now stay configurable.
    INPUT_MAT_FILE = "my_data.mat"
    INPUT_MAT_VAR = "data_in_matlab"

    out_path, out_var = model(INPUT_MAT_FILE, INPUT_MAT_VAR)
    print(f"Saved processed data to: {out_path} as variable '{out_var}'")