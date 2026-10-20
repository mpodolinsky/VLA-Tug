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
    """Mean and std over all steps of the training episodes (std floored to avoid division by zero)."""
    raise NotImplementedError


def standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raise NotImplementedError
