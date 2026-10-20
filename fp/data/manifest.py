"""Episode table: one row per rollout (outcome, task, onset step, rollout length).

The onset is defined here, once. Every other module takes ``Episode.onset`` as given.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    key: str  # rollout directory name, the key into features.npz
    success: bool
    task: str
    n_steps: int  # policy steps in the rollout
    onset: int | None  # policy step of the failure onset; None for successes
    condition: str = ""  # stratification label: scene variant (LIBERO-10) or level (LIBERO-X)
    skill: str | None = None  # LIBERO-X only
    signature: str | None = None  # LIBERO-X only


def load_manifest(dataset: str) -> list[Episode]:
    """Read data/<dataset>/manifest.csv and labels. Onset = labelled failure frame mapped to a
    policy step and clamped to [0, n_steps - 1]."""
    raise NotImplementedError
