"""Load pooled features and standardise them with training-split statistics."""

import numpy as np

from ..config import FEATURE_DIM, get


def load_features(dataset: str, keys: list[str]) -> dict[str, np.ndarray]:
    """{episode key: (T, 6144) float32} for the given keys, from data/<dataset>/features.npz."""
    path = get(dataset).data_dir / "features.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with np.load(path) as z:
        absent = [k for k in keys if k not in z.files]
        if absent:
            raise KeyError(f"{len(absent)} episodes are not in {path.name}, e.g. {absent[:3]}")
        out = {k: z[k] for k in keys}
    for k, x in out.items():
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or x.dtype != np.float32:
            raise ValueError(f"features of {k} have shape {x.shape} and dtype {x.dtype}; expected (T, {FEATURE_DIM}) float32")
    return out


def fit_standardiser(train: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over all steps of the training episodes (float64 sums; 1e-6 is added to the std)."""
    total = square = 0.0
    steps = 0
    for x in train:
        x = x.astype(np.float64)
        total, square, steps = total + x.sum(axis=0), square + (x**2).sum(axis=0), steps + len(x)
    mean = total / steps
    return mean, np.sqrt(np.maximum(square / steps - mean**2, 0.0)) + 1e-6


def standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)
