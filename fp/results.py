"""Where results go and how they are read back, so plots never need to re-run anything.

    results/<dataset>/<split>/<method>/
        scores.npz            per-step failure probabilities of every episode of the dataset (key = episode key)
        weights.pt            learned parameters (every trained method)
        mode_logits.npz       probes with a mode head: (T, 10) logits per step of every episode
        semantic.npz          probes with a CLIP head: (T, 64) raw semantic vectors per step of every episode
        semantic_space.npz    the CLIP head's target space (``target`` (P, 64) per phrase, ``bank_index`` (P,)), to interpret semantic.npz
        history.json         what the run was: configs, per-term learning curves, best epoch, seconds, split hash, code version
        curve_<calibrator>.json   rows: alpha, fpr, tpr, tpr_early, on_time, ldr
        summary.json          area_to_fpr40, extended flag, LTT point, config, seed
    results/<dataset>/skipped.json    methods that could not run, with the reason

A run directory is written under a temporary name and renamed when complete, so a directory that exists is a finished run and a killed
job leaves nothing that looks finished. An existing run is never overwritten.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import RESULTS
from .data.splits import Split


def run_dir(dataset: str, split: str, method: str, root: Path = RESULTS) -> Path:
    return Path(root) / dataset / split / method


def is_done(path: Path) -> bool:
    return (Path(path) / "history.json").exists()


def write_run(path: Path, scores: dict[str, np.ndarray], info: dict, save_weights: Callable[[str], None] | None = None,
              extras: Mapping[str, Mapping[str, np.ndarray]] | None = None) -> None:
    """Write scores, history, weights (if a saver is given) and any extra arrays files (``extras``: file name -> arrays) into ``path``, atomically."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; runs are never overwritten")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.savez(partial / "scores.npz", **scores)
    for name, arrays in (extras or {}).items():
        np.savez(partial / f"{name}.npz", **arrays)
    if save_weights is not None:
        save_weights(str(partial / "weights.pt"))
    (partial / "history.json").write_text(json.dumps(info))
    os.replace(partial, path)


def read_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path) / "scores.npz") as z:
        return {k: z[k] for k in z.files}


def read_history(path: Path) -> dict:
    return json.loads((Path(path) / "history.json").read_text())


def split_hash(split: Split) -> str:
    """Identifies the exact episodes of a split, so a result can be matched to the split file it was trained on."""
    digest = hashlib.sha256()
    for part in (split.train, split.val, split.calibration, split.seen, split.unseen):
        digest.update(("\n".join(part) + "\x00").encode())
    return digest.hexdigest()


def code_version() -> dict:
    """The git commit and a hash of the source of ``fp/``, which also covers changes that are not committed yet."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for p in sorted((root / "fp").rglob("*.py")):
        digest.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_head": head.stdout.strip() or None, "fp_sha256": digest.hexdigest()}


def write_curve(path: Path, rows: list[dict]) -> None:
    raise NotImplementedError


def read_curves(dataset: str, method: str, calibrator: str) -> dict[str, list[dict]]:
    """{split name: rows} for plotting."""
    raise NotImplementedError
