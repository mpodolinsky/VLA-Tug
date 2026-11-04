"""The embedding spaces on the left of the figure: the first three principal axes of three layers of the probe (``cli.tsne --reducer pca``, axes fitted on the
pre-failure steps of the training episodes). The page draws, per layer, a sample of the training points, one Gaussian per class and the path of the chosen episode."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fp.data.splits import Split
from fp.eval import class_fits, tsne

WINDOW = "pre"  # the steps up to the failure step (all steps of a success), as in the appendix figures
POINTS_PER_CLASS = 500


@dataclass(frozen=True)
class Layer:
    name: str  # the key on the page
    embedding: str  # the folder prefix of the saved points
    kind: str  # the classes: 'outcome' (success / failure) or 'mode' (the failure modes)
    failures_only: bool
    title: str


LAYERS = (
    Layer("lstm", "probe", "outcome", False, "LSTM state"),
    Layer("score_head", "probe_score_hidden", "outcome", False, "Failure score head"),
    Layer("mode_head", "probe_mode_hidden", "mode", True, "Failure mode head"),  # failures only, as in the appendix
)


def points_path(root: Path, dataset: str, split: str, method: str, layer: Layer) -> Path:
    return root / "pca3_train" / dataset / f"{layer.embedding}__{split}__{method}" / "points.npz"


def gaussian(mean: np.ndarray, cov: np.ndarray) -> tuple[list, list]:
    """The mean and the axes of the one-standard-deviation ellipsoid (ellipse in two dimensions): the page draws mean + axes @ (unit sphere)."""
    values, vectors = np.linalg.eigh(cov)
    axes = (vectors * class_fits.RADIUS * np.sqrt(np.maximum(values, 0))).T  # one row per axis
    return np.round(mean, 3).tolist(), np.round(axes, 3).tolist()


def summary(points: dict, split: Split, layer: Layer, seed: int) -> dict:
    """The training points of the layer's classes (a random sample of each) and the Gaussians of the classes, fitted on all their training points."""
    train = tsne.train_mask(list(points["keys"]), split)[points["episode"]]
    fits = [f for f in class_fits.fit_classes(points, layer.kind, train, WINDOW, columns="xyz") if not (layer.failures_only and f.id == 0)]
    rng = np.random.default_rng(seed)
    classes, xyz, labels = [], [], []
    for index, fit in enumerate(fits):
        mean3, axes3 = gaussian(fit.mean, fit.cov)
        mean2, axes2 = gaussian(fit.mean[:2], fit.cov[:2, :2])  # the marginal of the Gaussian in the first two axes
        classes.append({"name": fit.name, "episodes": fit.episodes, "mean": mean3, "axes": axes3, "mean2": mean2, "axes2": axes2})
        chosen = rng.choice(fit.rows, size=min(POINTS_PER_CLASS, len(fit.rows)), replace=False)
        xyz.append(points["xyz"][chosen])
        labels.append(np.full(len(chosen), index))
    return {
        "layer": layer.name,
        "title": layer.title,
        "window": WINDOW,
        "explained_variance": np.round(np.asarray(points["config"].get("explained_variance", [])), 4).tolist(),
        "classes": classes,
        "points": np.round(np.concatenate(xyz), 3).tolist(),
        "point_class": np.concatenate(labels).tolist(),
    }


def episode_path(points: dict, key: str) -> np.ndarray:
    """(steps, 3): the episode's states in the three principal axes, in the order of its steps."""
    index = list(points["keys"]).index(key)
    rows = np.flatnonzero(points["episode"] == index)
    return points["xyz"][rows[np.argsort(points["step"][rows])]]
