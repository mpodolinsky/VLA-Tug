"""One-off prepare step: raw rollout tokens -> per-modality pooled features.

This is the only module that reads the raw ``rollout.npz`` files. Output is
``data/<dataset>/features.npz`` with one float32 array (n_steps, 6144) per episode key,
laid out as [base | wrist | language].
"""

import numpy as np


def pool_modalities(base: np.ndarray, wrist: np.ndarray, language: np.ndarray, language_mask: np.ndarray) -> np.ndarray:
    """Masked mean over tokens, per modality.

    base, wrist: (T, tokens, 2048); language: (T, tokens, 2048); language_mask: (T, tokens) bool.
    Returns (T, 6144). Image tokens are all valid; language tokens are averaged where mask is True.
    """
    raise NotImplementedError


def prepare_dataset(dataset: str, rollout_root: str) -> None:
    """Pool every rollout of ``dataset`` found under ``rollout_root`` and write features.npz."""
    raise NotImplementedError
