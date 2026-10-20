"""Evaluate one (dataset, split, method): score -> calibrate -> sweep -> store rows.

Main table: SAFE band for SAFE models, SAFE band and max-conformal for ours, max-conformal for
STAC and the distance baselines. LTT is a separate evaluation (``evaluate_ltt``).
Methods that cannot be built raise NotImplementedError and are returned as skipped.
"""

from ..config import DatasetSpec
from ..data.splits import Split


def evaluate_method(spec: DatasetSpec, split: Split, method: str, calibrator: str) -> dict:
    """Rows of the alpha sweep and the headline area for one method and calibrator."""
    raise NotImplementedError


def evaluate_ltt(spec: DatasetSpec, split: Split, method: str) -> dict:
    """Single operating point selected by LTT: tpr, tpr_early, on_time, ldr, fpr."""
    raise NotImplementedError


def run_all(spec: DatasetSpec, methods: list[str]) -> dict:
    """Every split x method x calibrator; skipped methods are listed with their reason."""
    raise NotImplementedError
