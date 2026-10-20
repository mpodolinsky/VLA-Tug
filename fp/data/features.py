"""Load pooled features and standardise them with training-split statistics."""

import numpy as np


def load_features(dataset: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Return {episode key: (T, 6144) float32} from data/<dataset>/features.npz."""
    raise NotImplementedError


def fit_standardiser(train: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over all steps of the training episodes (std floored to avoid division by zero)."""
    raise NotImplementedError


def standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raise NotImplementedError
