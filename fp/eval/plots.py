"""The curve figures: TPR early (or on-time TPR) against FPR, one panel per setting in a single row, one line per method, the legend below.

Our probes (one line per label variant) are drawn thick in four colours sampled from viridis; every baseline is thin and neutral: SAFE black,
the action-signal baselines dark gray, the embedding-distance baselines light gray. Methods that share a colour are told apart by line style, and
ours stand out by width as well as by hue, because a mid gray can look like a viridis blue or teal to a colour-blind reader (the viridis
samples are separated by a colour-blind distance of 14.7 and a normal-vision one of 16.4, checked with a port of the dataviz skill's
palette validator). Axis labels, ticks and spines are black, only horizontal grid lines are drawn, and the legend has no category headings.
The text uses matplotlib's default font (DejaVu Sans); the PDF embeds it as TrueType.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..metrics.curves import FPR_MAX  # noqa: E402
from .figures import GRID, Entry, Setting  # noqa: E402

VIRIDIS_AT = {"onset_pre": 0.55, "onset_traj": 0.10, "onset": 0.33, "onset_post": 0.75}  # positions on the viridis map, one per label variant
NEUTRAL = {"safe": "#000000", "action": "#595959", "distance": "#a6a6a6"}
STYLES = {"solid": "-", "dashed": (0, (5, 2)), "dotted": (0, (1, 1.5)), "dashdot": (0, (5, 2, 1, 2))}
INK, GRID_COLOR = "#000000", "#e6e5e1"
WIDTH = {"ours": 2.4}
DEFAULT_WIDTH = 1.1
PANEL_WIDTH, PANEL_HEIGHT = 2.5, 2.9  # inches per panel
LEGEND_INCHES = 0.5  # height reserved below the panels for the legend


def color_of(entry: Entry):
    return matplotlib.colormaps["viridis"](VIRIDIS_AT[entry.variant]) if entry.kind == "ours" else NEUTRAL[entry.kind]


def style_axes() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
                         "grid.color": GRID_COLOR, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})  # 42: the font is embedded as TrueType, not as outlines of Type 3 glyphs


def line_kwargs(entry: Entry) -> dict:
    return dict(color=color_of(entry), linestyle=STYLES[entry.style], linewidth=WIDTH.get(entry.kind, DEFAULT_WIDTH), solid_capstyle="round")


def curves_figure(rows: list[tuple[str, dict[tuple[str, str], list[float]]]], settings: list[Setting], entries: list[Entry], ylabel: str,
                  panel_width: float = PANEL_WIDTH, legend_rows: int = 2, scale: float = 1.0) -> Figure:
    """``rows`` is a list of (row label, curves), top to bottom; ``curves[(setting label, entry label)]`` is the mean curve on ``GRID`` (percent)
    and entries or settings without one are left out. One panel per setting in each row, the setting names above the top row, the FPR label
    below the bottom one, the row label with ``ylabel`` on the left; the legend below, one entry per method drawn (in the order of ``entries``).
    Each panel is ``panel_width`` inches wide (a figure with fewer settings can keep the width of a full one by making its panels wider); the legend
    has ``legend_rows`` rows. ``scale`` shrinks the figure (the panels and the room for the legend, so the height to width ratio is kept) but not the
    text, which is then larger relative to the figure; below 1 the legend entries are set closer together so that a single row still fits."""
    style_axes()
    n_rows = len(rows)
    height = PANEL_HEIGHT * n_rows * scale
    fig, axes = plt.subplots(n_rows, len(settings), figsize=(panel_width * len(settings) * scale, height), squeeze=False, sharex="col")
    drawn = []
    for r, ((row_label, curves), row_axes) in enumerate(zip(rows, axes)):
        for ax, setting in zip(row_axes, settings):
            for entry in sorted(entries, key=lambda e: e.kind == "ours"):  # ours on top
                y = curves.get((setting.label, entry.label))
                if y is None:
                    continue
                ax.plot(GRID, y, **line_kwargs(entry), zorder=3 if entry.kind == "ours" else 2)
                if entry not in drawn:
                    drawn.append(entry)
            ax.set_xlim(0, FPR_MAX)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(setting.label)
            if r == n_rows - 1:
                ax.set_xlabel("FPR (%)")
        row_axes[0].set_ylabel(f"{row_label}\n{ylabel}")
    shown = [e for e in entries if e in drawn]
    fig.tight_layout(rect=(0, LEGEND_INCHES * scale / height, 1, 1))
    spacing = dict(fontsize=9, handlelength=2.8, columnspacing=1.6) if scale >= 1 else dict(fontsize=8.5, handlelength=1.8, columnspacing=0.9, handletextpad=0.5)
    fig.legend([Line2D([], [], **line_kwargs(e)) for e in shown], [e.label for e in shown], loc="lower center", ncol=max(1, math.ceil(len(shown) / legend_rows)),
               labelspacing=0.5, bbox_to_anchor=(0.5, 0.0), **spacing)
    return fig


def save(fig: Figure, stem: str, out_dir: Path) -> list[Path]:
    """Write ``stem``.png (200 dpi) and ``stem``.pdf into ``out_dir`` and close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths
