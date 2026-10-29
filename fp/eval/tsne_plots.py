"""The figures of the saved t-SNE points (``fp.eval.tsne``): one PNG per colouring and subset, all on the same axes, so that nothing here repeats the t-SNE.

Colourings (every point is one policy step of an episode):

    task      one hue per task (no legend: LIBERO-X has 218 tasks)
    skill     the skill of the episode's task (datasets with skill folds)
    outcome   successes go from light blue at the first step to dark blue at the last; failures from light blue to red at the failure step and stay red after it
    mode      as outcome for the successes; a failure goes from light blue to the colour of its failure mode, reached at the failure step
              (``MODE_COLORS``, one colour per class of ``fp.data.targets.MODE_NAMES``) and kept after it

Line plots (``outcome_lines``, ``mode_lines``): a sample of the episodes of the subset drawn as lines through their points, in the colours of the outcome or the mode colouring,
with a ring at the first step, over the other points of the subset in light grey: the progression of an episode can be followed (not for one point per episode).

Ellipse plots (``outcome_ellipses_late``, ``_all``, ``_post`` and ``_pre``, ``mode_ellipses_late``, ``_all``, ``_post`` and ``_pre``): the points of the subset in light grey, those of each class
(success and failure, or success and each failure mode) in its colour, and the one-standard-deviation ellipse of the Gaussian fitted to them with its centre (``fp.eval.class_fits``), for the late half
of each episode (up to the failure step), for all steps, or for the steps after the failure step (``post``; the successes keep the late half), or for the steps up to the failure step (``pre``; the successes keep all steps).

Pair plots (``outcome_ellipses_pairs``, drawn only when asked for): ``N_PAIRS`` figures per subset, ``outcome_ellipses_pair1`` to ``outcome_ellipses_pair5``, each the ellipse plot of the
outcome with the paths of a random success and a random failure of any mode (``fp.eval.class_fits.random_pairs``, the same episodes for the same subset and seed), to pick good-looking
trajectories.

Trajectory plots (``outcome_ellipses_path``, ``mode_ellipses_path``): the ellipse plot of the late half with the path of one successful episode drawn as a black arrow, smoothed over
``SMOOTH_STEPS`` steps, with a ring at its first step: the most typical successful episode of the subset (``fp.eval.class_fits.typical_episode``). The ``_paths`` versions
(``outcome_ellipses_paths``, ``mode_ellipses_paths``) add the path of the most typical failure of mode ``PATH_MODE``, dashed, up to its failure step, ending in a cross.

Gaussian plots without the points (``outcome_gaussians_late``, ..., ``mode_gaussians_all``): the same ellipses and centres without any points, on the axes of the other plots.
Together with the colourings (the points) and the ellipse plots (the points and the Gaussians) they are the versions without, with and with only the Gaussians.

3D plots (``outcome_3d``, ``mode_3d``; PCA points only, which keep a third axis): the colours of the outcome or mode colouring on the first three principal axes (labelled PC1, PC2,
PC3 with the share of the variance each holds, as the arrows of the 2D plots of PCA points), seen from three directions side by side, for a random sample of ``THREE_D_POINTS`` points of the subset.

3D Gaussians (``outcome_3d_ellipsoids``, ``outcome_3d_gaussians``, ``mode_3d_ellipsoids``, ``mode_3d_gaussians``): a Gaussian fitted to each class in the space of the three axes
(the window of the name: ``late`` without a suffix, or ``pre``), drawn as the one-standard-deviation ellipsoid with its centre, over the points (``_ellipsoids``) or alone
(``_gaussians``).

Subsets (``fp.eval.tsne.subset_masks``): all, seen, seen_held_out, unseen, each a mask over the same points. The axis limits are those of all the points.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..data.inputs import load_inputs
from ..data.targets import MODE_NAMES
from . import class_fits
from .tsne import SUBSETS, load_points, progress, subset_masks

LIGHT_BLUE, DARK_BLUE, RED = np.array([0.80, 0.88, 0.96]), np.array([0.03, 0.19, 0.42]), np.array([0.84, 0.15, 0.15])
MODE_COLORS = ("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#000000")  # one per mode, none of them blue
SKILL_COLORS = ("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf", "#9467bd", "#e377c2", "#8c564b")  # the skills, in alphabetical order
SUBSET_TITLES = {"all": "all episodes", "seen": "seen tasks", "seen_held_out": "seen tasks, held out", "unseen": "unseen tasks"}
POINT_SIZE, DPI, SIZE = 0.8, 200, 6.5  # marker area (points squared), resolution and edge of the square figure (inches)
RAMP_Y = 0.05  # the bottom of the colour ramps, as a fraction of the figure height
MODE_LEGEND_Y, MODE_LEGEND_STEP = 0.185, 0.03  # the first row of the mode legend and the distance between rows, as fractions of the figure height
BOTTOM = {"task": 0.02, "skill": 0.09, "outcome": 0.11, "mode": 0.23, "outcome_ellipses_late": 0.08, "outcome_ellipses_all": 0.08, "mode_ellipses_late": 0.2, "mode_ellipses_all": 0.2,
          "outcome_gaussians_late": 0.08, "outcome_gaussians_all": 0.08, "mode_gaussians_late": 0.2, "mode_gaussians_all": 0.2, "outcome_ellipses_path": 0.09, "mode_ellipses_path": 0.21, "outcome_ellipses_paths": 0.1, "mode_ellipses_paths": 0.24, "outcome_ellipses_post": 0.08, "mode_ellipses_post": 0.2, "outcome_ellipses_pre": 0.08, "mode_ellipses_pre": 0.2}  # the bottom of the plot of each colouring: room for its legend and ramps
LINES = {"outcome_lines": "outcome", "mode_lines": "mode"}  # the line plots and the colouring whose colours they use
LINE_SUCCESSES, LINE_FAILURES, LINE_WIDTH, BACKGROUND = 15, 25, 1.2, (0.88, 0.88, 0.88)  # episodes drawn as lines per figure, their width and the colour of the other points
ELLIPSES = {"outcome_ellipses_late": ("outcome", "late", True), "outcome_ellipses_all": ("outcome", "all", True), "mode_ellipses_late": ("mode", "late", True),
            "mode_ellipses_all": ("mode", "all", True), "outcome_gaussians_late": ("outcome", "late", False), "outcome_gaussians_all": ("outcome", "all", False),
            "mode_gaussians_late": ("mode", "late", False), "mode_gaussians_all": ("mode", "all", False),
            "outcome_ellipses_post": ("outcome", "post", True), "mode_ellipses_post": ("mode", "post", True),
            "outcome_ellipses_pre": ("outcome", "pre", True), "mode_ellipses_pre": ("mode", "pre", True)}  # (kind of class, window, with the points)
PATHS = {"outcome_ellipses_path": ("outcome", "late", False), "mode_ellipses_path": ("mode", "late", False), "outcome_ellipses_paths": ("outcome", "late", True),
         "mode_ellipses_paths": ("mode", "late", True)}  # the ellipse plots with the path of a successful episode and, if True, of a failure: (kind of class, window, with the failure)
PATH_MODE = "grasp_failure"  # the failure mode of the failure whose path is drawn
PAIRS = {"outcome_ellipses_pairs": ("outcome", "late")}  # random pairs of a success and a failure of any mode on the ellipse plot, one figure per pair (not drawn by default)
N_PAIRS = 5  # pairs drawn per subset
LAYER_TITLES = {"hidden": "state", "mode_hidden": "mode-head layer (64-d)", "score_hidden": "score-head layer (64-d)", "clip": "CLIP head (64-d)"}  # what the title calls the layer of the probe that is embedded
SMOOTH_STEPS = 7  # the path of an episode is drawn averaged over this many steps
CAPTION_CHARS = 80  # the caption naming the task of a path is cut to this many characters
WINDOW_TITLES = {"late": "second half of the steps before the failure", "all": "all steps", "post": "steps after the failure step", "pre": "steps up to the failure step"}
THREE_D = {"outcome_3d": "outcome", "mode_3d": "mode"}  # the 3D plots and the colouring whose colours they use
VIEWS = ((20, -60), (20, 30), (20, 120))  # (elevation, azimuth) in degrees of the three panels of a 3D plot
THREE_D_POINTS = 40_000  # points drawn per panel of a 3D plot (a random sample of the subset)
THREE_D_FITS = {"outcome_3d_ellipsoids": ("outcome", True, "late"), "outcome_3d_gaussians": ("outcome", False, "late"), "mode_3d_ellipsoids": ("mode", True, "late"),
                "mode_3d_gaussians": ("mode", False, "late"), "outcome_3d_ellipsoids_pre": ("outcome", True, "pre"), "outcome_3d_gaussians_pre": ("outcome", False, "pre"),
                "mode_3d_ellipsoids_pre": ("mode", True, "pre"), "mode_3d_gaussians_pre": ("mode", False, "pre")}  # (kind of class, with the points, window of the fit)
LIMIT_PERCENTILE, LIMIT_PAD = 0.1, 0.05  # the axes cover the 0.1 to 99.9 percentiles of the points, plus 5% on each side


def mix(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(N, 3) colours going from ``start`` (t = 0) to ``end`` (t = 1) in RGB; ``end`` may be one colour or one per row."""
    return start + (np.asarray(end) - start) * t[:, None]


def by_task(points: dict) -> np.ndarray:
    """One colour per task: hues spread by the golden angle, in the order of the task names, so that neighbouring names differ."""
    tasks, index = np.unique(points["task"], return_inverse=True)
    hue = (np.arange(len(tasks)) * 0.61803398875) % 1.0
    value = 0.95 - 0.2 * ((np.arange(len(tasks)) // 7) % 3) / 2
    table = mcolors.hsv_to_rgb(np.stack([hue, np.full(len(tasks), 0.7), value], axis=1))
    return table[index][points["episode"]]


def skills_of(points: dict) -> list[str]:
    return sorted(set(points["skill"].tolist()) - {""})


def by_skill(points: dict) -> np.ndarray:
    names = skills_of(points)
    table = np.array([mcolors.to_rgb(SKILL_COLORS[i]) for i in range(len(names))] + [(0.7, 0.7, 0.7)])
    index = np.array([names.index(s) if s in names else len(names) for s in points["skill"]])
    return table[index][points["episode"]]


def by_outcome(points: dict) -> np.ndarray:
    ep, t = points["episode"], progress(points)
    return np.where(points["success"][ep][:, None], mix(LIGHT_BLUE, DARK_BLUE, t), mix(LIGHT_BLUE, RED, t))


def by_mode(points: dict) -> np.ndarray:
    ep, t = points["episode"], progress(points)
    table = np.array([mcolors.to_rgb(c) for c in MODE_COLORS])
    failure = mix(LIGHT_BLUE, table[np.maximum(points["mode"][ep], 0)], t)
    return np.where(points["success"][ep][:, None], mix(LIGHT_BLUE, DARK_BLUE, t), failure)


COLORINGS = {"task": by_task, "skill": by_skill, "outcome": by_outcome, "mode": by_mode}
FIGURES = tuple(COLORINGS) + tuple(LINES) + tuple(ELLIPSES) + tuple(PATHS) + tuple(THREE_D) + tuple(THREE_D_FITS)  # everything ``render`` draws by default
BASE = LINES | THREE_D | {name: kind for name, (kind, _, _) in THREE_D_FITS.items()}  # the colouring each line plot and 3D plot takes its colours from
NEEDS_XYZ = tuple(THREE_D) + tuple(THREE_D_FITS)  # the figures that need the third principal axis


def legend_handles(coloring: str, points: dict) -> list:
    if coloring == "skill":
        return [Patch(color=SKILL_COLORS[i], label=name) for i, name in enumerate(skills_of(points))]
    return []


def mode_entries(points: dict) -> list[tuple[str, np.ndarray]]:
    """(label, the colour its ramp ends in) of the mode figures: the successes, then each failure mode that occurs."""
    used = sorted(set(points["mode"].tolist()) - {-1})
    return [("success", DARK_BLUE)] + [(MODE_NAMES[m], np.array(mcolors.to_rgb(MODE_COLORS[m]))) for m in used]


def add_mode_legend(fig, points: dict) -> None:
    """One small bar per entry of ``mode_entries``, light blue at the first step to the colour of the entry (success: at the last step, failure: at the failure step, constant after),
    with its label, in three columns under the plot."""
    for k, (label, end) in enumerate(mode_entries(points)):
        x, y = 0.04 + 0.32 * (k % 3), MODE_LEGEND_Y - MODE_LEGEND_STEP * (k // 3)
        bar = fig.add_axes([x, y, 0.06, 0.012])
        bar.imshow(mix(LIGHT_BLUE, end, np.linspace(0, 1, 64))[None], aspect="auto")
        bar.axis("off")
        fig.text(x + 0.07, y + 0.006, label, fontsize=7.5, va="center")
    fig.text(0.5, 0.012, "each point: light blue at the first step, the colour of its entry at the last step (success) or at the failure step (failure), constant after", fontsize=7, ha="center",
             color="0.35")


def add_ramps(fig, ax, coloring: str) -> None:
    """The colour ramps of the outcome colouring as two small bars under the plot."""
    ramps = [("success: first to last step", DARK_BLUE), ("failure: first step to failure step, then constant", RED)]
    for k, (label, end) in enumerate(ramps):
        bar = fig.add_axes([0.14 + 0.43 * k, RAMP_Y, 0.38, 0.014])
        bar.imshow(mix(LIGHT_BLUE, end, np.linspace(0, 1, 256))[None], aspect="auto")
        bar.set_xticks([])
        bar.set_yticks([])
        bar.set_xlabel(label, fontsize=7.5, labelpad=2)


def limits_of(xy: np.ndarray) -> tuple[tuple[float, float], ...]:
    """The axis limits of the points, one (low, high) per column: their 0.1 to 99.9 percentiles, padded."""
    out = []
    for d in range(xy.shape[1]):
        lo, hi = np.percentile(xy[:, d], [LIMIT_PERCENTILE, 100 - LIMIT_PERCENTILE])
        out.append((lo - LIMIT_PAD * (hi - lo), hi + LIMIT_PAD * (hi - lo)))
    return tuple(out)


def axis_labels(points: dict) -> tuple[str, ...] | None:
    """The labels of the axes of PCA points, 'PC1 (4.0%)': the principal axis and the share of the variance it holds; None for other points (t-SNE axes mean nothing)."""
    config = points.get("config", {})
    if config.get("reducer") != "pca":
        return None
    shares = config.get("explained_variance") or [None] * 3
    return tuple(f"PC{i + 1}" + ("" if v is None else f" ({100 * v:.1f}%)") for i, v in enumerate(shares))


def add_axes_arrows(ax, labels: tuple[str, ...]) -> None:
    """A small coordinate system in the lower left corner of a 2D plot: an arrow to the right labelled with the first axis and one upwards with the second."""
    arrow = dict(arrowstyle="-|>", color="0.25", linewidth=1.2, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0.16, 0.03), xytext=(0.03, 0.03), xycoords="axes fraction", arrowprops=arrow)
    ax.annotate("", xy=(0.03, 0.16), xytext=(0.03, 0.03), xycoords="axes fraction", arrowprops=arrow)
    ax.text(0.16, 0.045, labels[0], transform=ax.transAxes, fontsize=8, color="0.25", ha="left", va="bottom")
    ax.text(0.045, 0.16, labels[1], transform=ax.transAxes, fontsize=8, color="0.25", ha="left", va="bottom")


def canvas(coloring: str, title: str, limits, labels: tuple[str, ...] | None = None):
    """A square figure without axes, titled, with room below for the legend and the ramps of ``coloring``, and a small coordinate system if there are ``labels``."""
    fig, ax = plt.subplots(figsize=(SIZE, SIZE))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=BOTTOM[coloring])
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    if labels:
        add_axes_arrows(ax, labels)
    return fig, ax


def draw(points: dict, colors: np.ndarray, mask: np.ndarray, coloring: str, title: str, path: Path, order: np.ndarray, limits) -> None:
    """One PNG: the points of ``mask`` (a mask over points) in the colours ``colors``, in the fixed order ``order`` so that no episode is always on top."""
    fig, ax = canvas(coloring, title, limits, axis_labels(points))
    shown = order[mask[order]]
    ax.scatter(points["xy"][shown, 0], points["xy"][shown, 1], c=colors[shown], s=POINT_SIZE, linewidths=0, rasterized=True)
    finish(fig, ax, coloring, points, path)


def sample_episodes(episodes: np.ndarray, success: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Up to ``LINE_SUCCESSES`` successes and ``LINE_FAILURES`` failures, drawn at random from the episode indices ``episodes`` (sorted)."""
    wins, fails = episodes[success[episodes]], episodes[~success[episodes]]
    return np.sort(np.concatenate([rng.choice(a, min(n, len(a)), replace=False) for a, n in ((wins, LINE_SUCCESSES), (fails, LINE_FAILURES))]))


def trajectory_segments(points: dict, chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(segments (S, 2, 2), the point index each segment ends at, the point index each episode starts at): consecutive points of the same chosen episode, which the file keeps in the order
    of the steps; there is no segment between two episodes."""
    episode = points["episode"]
    selected = np.isin(episode, chosen)
    start = np.flatnonzero(selected[:-1] & (episode[:-1] == episode[1:]))
    first = np.flatnonzero(selected & np.concatenate([[True], episode[1:] != episode[:-1]]))
    return np.stack([points["xy"][start], points["xy"][start + 1]], axis=1), start + 1, first


def draw_lines(points: dict, colors: np.ndarray, mask: np.ndarray, chosen: np.ndarray, coloring: str, title: str, path: Path, limits) -> None:
    """One PNG: the points of ``mask`` in light grey and the ``chosen`` episodes as lines in the colours ``colors`` (each segment the colour of the point it ends at), with a ring at
    the first step of each."""
    fig, ax = canvas(coloring, title, limits, axis_labels(points))
    ax.scatter(points["xy"][mask, 0], points["xy"][mask, 1], color=BACKGROUND, s=POINT_SIZE, linewidths=0, rasterized=True)
    segments, ends, first = trajectory_segments(points, chosen)
    ax.add_collection(LineCollection(segments, colors=colors[ends], linewidths=LINE_WIDTH, capstyle="round"))
    ax.scatter(points["xy"][first, 0], points["xy"][first, 1], facecolors="white", edgecolors="black", s=10, linewidths=0.6, zorder=3)
    finish(fig, ax, coloring, points, path)


def draw_3d(points: dict, colors: np.ndarray, mask: np.ndarray, coloring: str, title: str, path: Path, limits, rng: np.random.Generator, fits=None, kind: str | None = None,
            show_points: bool = True) -> None:
    """One PNG: a random sample of ``THREE_D_POINTS`` points of ``mask`` in the colours ``colors`` on the three principal axes (``points["xyz"]``), seen from the ``VIEWS``, with
    the legend or the ramps of ``coloring``; with ``fits`` (``fp.eval.class_fits``, of the classes of ``kind``) also the ellipsoid and the centre of each, and without
    ``show_points`` only those, with a legend of the classes."""
    fig = plt.figure(figsize=(3 * 5.0, 6.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=BOTTOM[coloring] if show_points else (0.17 if coloring == "mode" else 0.08), wspace=0.0)
    shown = rng.permutation(np.flatnonzero(mask))[:THREE_D_POINTS]
    xyz, labels = points["xyz"][shown], axis_labels(points)
    for k, (elevation, azimuth) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if show_points:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors[shown], s=POINT_SIZE, linewidths=0, depthshade=False, rasterized=True)
        for fit in fits or []:
            color = class_color(kind, fit.id)
            ax.plot_surface(*class_fits.ellipsoid(fit.mean, fit.cov), color=color, alpha=0.3, linewidth=0, antialiased=False, shade=False)
            ax.scatter(*fit.mean, color=color, s=40, edgecolors="white", linewidths=0.8, depthshade=False)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_box_aspect([hi - lo for lo, hi in limits])
        ax.view_init(elev=elevation, azim=azimuth)
        if labels:
            ax.set_xlabel(labels[0], fontsize=9, labelpad=-6)
            ax.set_ylabel(labels[1], fontsize=9, labelpad=-6)
            ax.set_zlabel(labels[2], fontsize=9, labelpad=-6)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.tick_params(length=0)
        else:
            ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    if show_points:
        finish(fig, ax, coloring, points, path)
    else:
        add_class_legend(fig, kind, fits or [], 4)
        save(fig, path)


def class_color(kind: str, class_id: int) -> np.ndarray:
    """The colour of a class of ``fp.eval.class_fits``: dark blue for the successes, red for the failures (outcome) or the colour of the failure mode."""
    if class_id == 0:
        return DARK_BLUE
    return RED if kind == "outcome" else np.array(mcolors.to_rgb(MODE_COLORS[class_id - 1]))


def smoothed(xy: np.ndarray, window: int = SMOOTH_STEPS) -> np.ndarray:
    """The rows of ``xy`` averaged over a centred window of ``window`` rows (fewer at both ends, and never more than the rows there are): the same number of rows."""
    half = min(window // 2, (len(xy) - 1) // 2)
    kernel = np.ones(2 * half + 1)
    counts = np.convolve(np.ones(len(xy)), kernel, mode="same")
    return np.stack([np.convolve(xy[:, d], kernel, mode="same") / counts for d in range(xy.shape[1])], axis=1)


def add_path(ax, points: dict, episode: int, failure: bool = False, caption_row: int = 0, label: str = "successful") -> None:
    """The path of the episode with index ``episode`` as a black arrow: a line smoothed over ``SMOOTH_STEPS`` steps with arrowheads along it and a ring at its first step; a failure
    (``failure``) is drawn dashed up to its failure step and ends in a cross. Its task is a small caption in the top left corner, in row ``caption_row``."""
    rows = np.flatnonzero(points["episode"] == episode)
    if failure:
        rows = rows[points["step"][rows] <= points["onset"][episode]]
    path = smoothed(points["xy"][rows].astype(np.float64))
    ax.plot(path[:, 0], path[:, 1], color="black", linewidth=2.0, linestyle="--" if failure else "-", zorder=6, solid_capstyle="round")
    for i in np.unique(np.linspace(1, len(path) - 1, 6).astype(int)):
        ax.annotate("", xy=path[i], xytext=path[i - 1], arrowprops=dict(arrowstyle="-|>", color="black", linewidth=0, mutation_scale=16, shrinkA=0, shrinkB=0), zorder=7)
    ax.scatter(*path[0], facecolors="white", edgecolors="black", s=40, linewidths=1.2, zorder=8)
    if failure:
        ax.scatter(*path[-1], marker="X", color="black", s=70, edgecolors="white", linewidths=0.8, zorder=9)
    caption = f"{label} episode: {points['task'][episode]}"
    ax.text(0.01, 0.99 - 0.035 * caption_row, caption if len(caption) <= CAPTION_CHARS else caption[: CAPTION_CHARS - 1] + "…", transform=ax.transAxes, fontsize=7, color="0.35", ha="left", va="top")


def add_class_legend(fig, kind: str, fits: list, ncol: int, extra=()) -> None:
    """A legend of the fitted classes under the plot: a swatch in the colour of each and its number of episodes, then the handles of ``extra``."""
    handles = [Patch(color=class_color(kind, f.id), label=f"{f.name} ({f.episodes} episodes)") for f in fits] + list(extra)
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=ncol, fontsize=7.5, frameon=False, handlelength=1.0, columnspacing=1.2)


def draw_ellipses(points: dict, mask: np.ndarray, kind: str, window: str, figure: str, title: str, path: Path, limits, show_points: bool = True,
                  path_episode: int | None = None, failure_episode: int | None = None, failure_mode: str = "") -> list:
    """One PNG: the points of ``mask`` in light grey, the points of each fitted class in its colour (both left out without ``show_points``) with the one-standard-deviation ellipse
    of its Gaussian and its centre, and the path of the successful episode ``path_episode`` as a black arrow and that of the failure ``failure_episode`` (of ``failure_mode``) as a
    dashed one; returns the fits."""
    fig, ax = canvas(figure, title, limits, axis_labels(points))
    if show_points:
        ax.scatter(points["xy"][mask, 0], points["xy"][mask, 1], color=BACKGROUND, s=POINT_SIZE, linewidths=0, rasterized=True)
    fits = class_fits.fit_classes(points, kind, mask, window)
    for fit in fits:
        color = class_color(kind, fit.id)
        if show_points:
            ax.scatter(points["xy"][fit.rows, 0], points["xy"][fit.rows, 1], color=color, s=POINT_SIZE, alpha=0.3, linewidths=0, rasterized=True)
        outline = class_fits.ellipse(fit.mean, fit.cov)
        ax.fill(outline[:, 0], outline[:, 1], color=color, alpha=0.12 if show_points else 0.25, linewidth=0)
        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.8)
        ax.scatter(*fit.mean, color=color, s=45, edgecolors="white", linewidths=1.0, zorder=4)
    extra = []
    if path_episode is not None:
        add_path(ax, points, path_episode)
        extra.append(Line2D([0], [0], color="black", linewidth=2.0, label="one successful episode (ring: first step)"))
    if failure_episode is not None:
        add_path(ax, points, failure_episode, True, int(path_episode is not None), failure_mode)
        extra.append(Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label=f"one {failure_mode} episode (X: failure step)"))
    add_class_legend(fig, kind, fits, 2 if len(fits) + len(extra) > 3 else 3, extra)
    save(fig, path)
    return fits


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def finish(fig, ax, coloring: str, points: dict, path: Path) -> None:
    """The legend and the ramps of ``coloring`` (the colouring a line plot takes its colours from), then the file."""
    handles = legend_handles(coloring, points)
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=7.5, frameon=False, handlelength=1.0, columnspacing=1.2)
    if coloring == "outcome":
        add_ramps(fig, ax, coloring)
    if coloring == "mode":
        add_mode_legend(fig, points)
    save(fig, path)


def embedding_title(config: dict) -> str:
    name = "VLA embedding" if config["embedding"] == "vla" else f"probe {LAYER_TITLES[config.get('layer') or 'hidden']} (trained on {config['split']})"
    return name + (", PCA" if config.get("reducer") == "pca" else "") + (", one point per episode" if config.get("unit") == "episode" else "")


def render(points_path: Path, split: str | None = None, out_dir: Path | None = None, colorings=None, subsets=SUBSETS, seed: int = 0, path_mode: str = PATH_MODE,
           n_pairs: int = N_PAIRS) -> list[Path]:
    """Draw ``<out_dir>/<figure>__<subset>.png`` for every figure (a colouring or a line plot, ``FIGURES``) and subset of the points saved in ``points_path`` (default out_dir:
    ``figures/<split>/`` next to it). The episodes of a line plot are a sample that depends on the subset and ``seed`` only, so that ``outcome_lines`` and ``mode_lines`` draw the same.
    ``split`` is the split whose seen / held-out / unseen episodes the subsets are: the probe's own for a probe (the default), any for the VLA embedding. A figure the points
    cannot show (skill without skills, lines for one point per episode, 3D without a third axis) and a subset that is empty are skipped. Returns the files written."""
    points = load_points(points_path)
    split = split or points["config"]["split"]
    out_dir = out_dir or points_path.parent / "figures" / split
    masks = subset_masks(points["keys"].tolist(), load_inputs(points["config"]["dataset"]).splits[split])
    order = np.random.default_rng(seed).permutation(len(points["xy"]))
    limits, written = limits_of(points["xy"]), []
    limits3 = limits_of(points["xyz"]) if "xyz" in points else None
    for figure in colorings or FIGURES:
        if (figure == "skill" and not skills_of(points)) or (figure in tuple(LINES) + tuple(PATHS) + tuple(PAIRS) and points["config"].get("unit") == "episode") or (figure in NEEDS_XYZ and "xyz" not in points):
            continue
        coloring = BASE.get(figure, figure)
        colors = None if figure in ELLIPSES or figure in PATHS or figure in PAIRS else COLORINGS[coloring](points)
        for subset in subsets:
            episodes = masks[subset]
            if not episodes.any():
                continue
            head = f"{embedding_title(points['config'])}\n{SUBSET_TITLES[subset]}"
            title = f"{head} ({int(episodes.sum())} episodes)"
            path = out_dir / f"{figure}__{subset}.png"
            if figure in ELLIPSES:
                kind, window, show_points = ELLIPSES[figure]
                draw_ellipses(points, episodes[points["episode"]], kind, window, figure, f"{head}: 1σ ellipses{'' if show_points else ' only'}, {WINDOW_TITLES[window]}", path, limits,
                              show_points)
            elif figure in PAIRS:
                kind, window = PAIRS[figure]
                mask = episodes[points["episode"]]
                pairs = class_fits.random_pairs(points, mask, n_pairs, np.random.default_rng([seed, SUBSETS.index(subset)]))
                for k, (success, failure) in enumerate(pairs):
                    pair_path = out_dir / f"{figure.removesuffix('s')}{k + 1}__{subset}.png"
                    mode = MODE_NAMES[points["mode"][failure]]
                    draw_ellipses(points, mask, kind, window, "outcome_ellipses_paths", f"{head}: pair {k + 1} of {len(pairs)}, a success and a {mode} failure", pair_path, limits,
                                  True, success, failure, mode)
                    written.append(pair_path)
                continue
            elif figure in PATHS:
                kind, window, with_failure = PATHS[figure]
                mask = episodes[points["episode"]]
                success = class_fits.typical_episode(points, mask, "outcome", 0, window)
                failure = class_fits.typical_episode(points, mask, "mode", class_fits.mode_class_id(path_mode), window) if with_failure else None
                what = "a successful trajectory" if not with_failure else f"a success and a {path_mode} trajectory"
                draw_ellipses(points, mask, kind, window, figure, f"{head}: 1σ ellipses, {what}", path, limits, True, success, failure, path_mode)
            elif figure in THREE_D_FITS:
                kind, show_points, window = THREE_D_FITS[figure]
                mask = episodes[points["episode"]]
                draw_3d(points, colors, mask, coloring, f"{head.replace(chr(10), ': ')}: 1σ ellipsoids{'' if show_points else ' only'}, {WINDOW_TITLES[window]}", path, limits3,
                        np.random.default_rng([seed, SUBSETS.index(subset)]), class_fits.fit_classes(points, kind, mask, window, "xyz"), kind, show_points)
            elif figure in THREE_D:
                draw_3d(points, colors, episodes[points["episode"]], coloring, f"{head.replace(chr(10), ': ')} ({int(episodes.sum())} episodes)", path, limits3,
                        np.random.default_rng([seed, SUBSETS.index(subset)]))
            elif figure in LINES:
                chosen = sample_episodes(np.flatnonzero(episodes), points["success"], np.random.default_rng([seed, SUBSETS.index(subset)]))
                draw_lines(points, colors, episodes[points["episode"]], chosen, coloring, f"{head}: {len(chosen)} of {int(episodes.sum())} episodes as lines", path, limits)
            else:
                draw(points, colors, episodes[points["episode"]], coloring, title, path, order, limits)
            written.append(path)
    return written
