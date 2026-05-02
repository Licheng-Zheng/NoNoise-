from ssim import ssim_hsi
import numpy as np

def run_psnr(clean: np.ndarray, processed: np.ndarray, *, max_val: float = 1.0, **kwargs) -> float:
    """Calculates Peak Signal-to-Noise Ratio.

    Required:
        clean, processed: numpy arrays of same shape
    Optional (keyword-only):
        max_val: dynamic range of the signal (1.0 for [0,1] data, 255 for 8-bit)
    Extra kwargs are accepted and ignored to support a common call-site.
    """
    mse = np.mean((clean - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def run_ssim(
    clean: np.ndarray,
    processed: np.ndarray,
    *,
    window_size: int = 11,
    normalize: bool = True,
    device: str | None = None,
    **kwargs,
) -> float:
    """Wrapper for your imported SSIM function with flexible optional settings."""
    return ssim_hsi(clean, processed, window_size=window_size, normalize=normalize, device=device)

def onesies(*args, **kwargs): 
    return 1

def twosies(*args, **kwargs): 
    return 2 