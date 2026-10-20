"""Where results go and how they are read back, so plots never need to re-run anything.

    results/<dataset>/<split>/<method>/
        weights.pt            learned parameters (every trained method)
        scores.npz            per-step scores for every calibration/eval episode
        curve_<calibrator>.json   rows: alpha, fpr, tpr, tpr_early, on_time, ldr
        summary.json          area_to_fpr40, extended flag, LTT point, config, seed
    results/<dataset>/skipped.json    methods that could not run, with the reason
"""

from pathlib import Path


def run_dir(dataset: str, split: str, method: str) -> Path:
    raise NotImplementedError


def write_curve(path: Path, rows: list[dict]) -> None:
    raise NotImplementedError


def read_curves(dataset: str, method: str, calibrator: str) -> dict[str, list[dict]]:
    """{split name: rows} for plotting."""
    raise NotImplementedError
