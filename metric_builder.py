from typing import Callable, Dict, Any, List

import numpy as np
import scipy.io as sio

# Import metric implementations
from Model_Evaluation.ssim import ssim_hsi


class MetricFactory:
    """
    A registry-based factory for creating metric callables by name.

    - register(name, creator): register a metric creator function.
    - create(name, **kwargs): create/return a metric callable for use.

    A metric callable has the signature metric(clean_hsi: np.ndarray, processed_hsi: np.ndarray) -> Any
    and can optionally expose a .metadata dict with metadata like name, description, etc.
    """

    def __init__(self) -> None:
        self._creators: Dict[str, Callable[..., Callable[[np.ndarray, np.ndarray], Any]]] = {}
        self._register_defaults()

    def register(self, name: str, creator: Callable[..., Callable[[np.ndarray, np.ndarray], Any]]) -> None:
        """Register a metric creator under a given name."""
        key = name.strip().lower()
        if not key:
            raise ValueError("Metric name must be a non-empty string")
        self._creators[key] = creator

    def create(self, name: str, **kwargs) -> Callable[[np.ndarray, np.ndarray], Any]:
        """
        Create a metric callable by name.
        kwargs are forwarded to the underlying creator to parametrize the metric.
        """
        key = name.strip().lower()
        if key not in self._creators:
            raise KeyError(f"Metric '{name}' is not registered. Registered metrics: {list(self._creators.keys())}")
        return self._creators[key](**kwargs)

    def available(self) -> Dict[str, Callable[..., Callable[[np.ndarray, np.ndarray], Any]]]:
        """Return a mapping of registered metric creators."""
        return dict(self._creators)

    def _register_defaults(self) -> None:
        """Register built-in metrics here."""
        # SSIM3D metric
        def make_ssim3d(window_size: int = 11, size_average: bool = True):
            def metric(clean_hsi: np.ndarray, processed_hsi: np.ndarray):
                return ssim_hsi(clean_hsi, processed_hsi, window_size=window_size, normalize=False)

            metric.metadata = {
                "name": "SSIM3D",
                "window_size": window_size,
                "size_average": size_average,
                "description": "3D Structural Similarity Index"
            }
            return metric

        self.register("ssim3d", make_ssim3d)

        def make_ssim_from_mat(
            mat_var: str = "cube",
            window_size: int = 11,
            normalize: bool = False,
        ):
            """
            Returns a callable metric that accepts two MAT paths and computes SSIM on (H,W,B) cubes.

            The callable signature will be metric_from_paths(clean_mat_path: str, processed_mat_path: str) -> float
            """
            def _load_mat_hsi(path: str) -> np.ndarray:
                data = sio.loadmat(path)
                if mat_var not in data:
                    raise KeyError(f"Variable '{mat_var}' not found in {path}. Available: {list(data.keys())}")
                cube = data[mat_var]
                if cube.ndim != 3:
                    raise ValueError(f"Expected 3D HSI (H,W,B). Got shape {cube.shape} from {path}")
                return cube

            def metric_from_paths(clean_mat_path: str, processed_mat_path: str) -> float:
                clean = _load_mat_hsi(clean_mat_path)
                proc = _load_mat_hsi(processed_mat_path)
                if clean.shape != proc.shape:
                    raise ValueError(f"HSI shapes differ: {clean.shape} vs {proc.shape}")
                return ssim_hsi(clean, proc, window_size=window_size, normalize=normalize)

            metric_from_paths.metadata = {
                "name": "SSIM3D_from_MAT",
                "window_size": window_size,
                "normalize": normalize,
                "mat_var": mat_var,
                "description": "3D SSIM for HSIs loaded from .mat files (expects (H,W,B))",
            }
            return metric_from_paths

        self.register("ssim_from_mat", make_ssim_from_mat)


class MetricTestSuite:
    """A simple suite that holds metric callables and runs them."""

    def __init__(self) -> None:
        self._tests: List[Callable[[np.ndarray, np.ndarray], Any]] = []

    def add(self, test_callable: Callable[[np.ndarray, np.ndarray], Any]) -> None:
        if not callable(test_callable):
            raise TypeError("Test must be callable")
        self._tests.append(test_callable)

    def run(self, clean_hsi: np.ndarray, processed_hsi: np.ndarray):
        results = []
        for test in self._tests:
            value = test(clean_hsi, processed_hsi)
            meta = getattr(test, "metadata", {"name": getattr(test, "__name__", "metric")})
            results.append({
                "name": meta.get("name", "metric"),
                "metadata": meta,
                "value": value,
            })
        return results


def build_default_suite() -> MetricTestSuite:
    """Build a default suite with built-in metrics (currently SSIM3D)."""
    factory = MetricFactory()
    suite = MetricTestSuite()

    # Add SSIM3D test with default params; customize as needed
    ssim_test = factory.create("ssim3d", window_size=11, size_average=True)
    suite.add(ssim_test)

    # To add future tests, register and add like:
    # factory.register("psnr", make_psnr)
    # suite.add(factory.create("psnr", max_val=1.0))

    return suite


# Backwards-compatible wrapper that mirrors your initial TestFactory usage
class TestFactory:
    def __init__(self, clean_hsi: np.ndarray, processed_hsi: np.ndarray):
        self.clean_hsi = clean_hsi
        self.processed_hsi = processed_hsi
        self._factory = MetricFactory()

    def create_ssim_test(self):
        metric = self._factory.create("ssim3d")
        return metric


def example_run_for_given_paths():
    """Run SSIM on the provided MAT paths using the factory-made loader metric."""
    factory = MetricFactory()

    mat_var = "data"  # change to your actual variable name inside the .mat files

    ssim_from_mat = factory.create(
        "ssim_from_mat",
        mat_var=mat_var,
        window_size=11,
        normalize=True,
    )

    processed_path = r"C:\\Users\\liche\\OneDrive\\Desktop\\PycharmProjects\\NoNoise-\\processed\\ksc512\\add.mat"
    clean_path = r"C:\\Users\\liche\\OneDrive\\Desktop\\PycharmProjects\\NoNoise-\\database\\ksc512\\clean.mat"

    score = ssim_from_mat(clean_path, processed_path)
    print({
        "metric": ssim_from_mat.metadata["name"],
        "value": score,
        "metadata": ssim_from_mat.metadata,
        "clean_path": clean_path,
        "processed_path": processed_path,
    })

example_run_for_given_paths()