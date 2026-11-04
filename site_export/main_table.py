"""The paper's main table as HTML for the project page: the mean of every cell with its standard deviation, and under each setting how many runs it averages.

    python -m site_export.main_table          # reads results/figures/table_main_std.tex, rewrites the table in website/index.html

``table_main_std.tex`` is written by ``python -m cli.figures`` (``table_main.tex`` is the same table without the standard deviations). The page takes the
paper's table itself, so that every value and every mark of the best and the second best value is the paper's (rounding included); this only changes
the markup. The number of runs of a setting is counted from the saved runs under ``results/``. The table lies between two comment markers of the
page; running this again replaces what is between them.
"""

import argparse
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fp.config import RESULTS
from fp.eval import figures
from fp.eval.select_config import group_of
from fp.results import iter_runs

START = "<!-- main-table:start (written by python -m site_export.main_table) -->"
END = "<!-- main-table:end -->"
OURS = "VLA-Tug (ours)"
# A qualitative label of how hard the held-out tasks of a setting are (more stars, harder); the real robot has none.
DIFFICULTY = {"LIBERO-10 π0.5": "*", "LIBERO-10 GR00T": "*", "LIBERO-X task": "*", "LIBERO-X signature": "**", "LIBERO-X skill": "***"}
CAPTION = ("Early failure detection on seen and unseen tasks: the area under the early-TPR vs. FPR curve up to an FPR of 40% (in %, higher is better), "
           "the mean over the runs of a setting, with the standard deviation over them after the &plusmn;; the number of runs and a qualitative difficulty (more stars, harder) are under the name of the setting "
           "(seeds; LIBERO-X skill: one fold per held-out skill; real robot: 5 seeds, each with one fold per held-out task, 8 folds). Seen: held-out episodes of "
           "training tasks; unseen: held-out tasks. Cal.: calibrator (MC max-conformal, FC functional conformal). Cost on LIBERO-X (lower is better): "
           "fit time, memory to score a new episode, and per-step latency on a GPU and a CPU. Dashes: not available. Best in bold, second best in italics.")


@dataclass(frozen=True)
class Cell:
    text: str | None  # None where the table has a dash
    sd: str | None = None  # the standard deviation, for the areas
    mark: str | None = None  # 'best', 'second' or None


@dataclass(frozen=True)
class Row:
    label: str
    ours: bool
    calibrator: str
    cells: list[Cell]


def unwrap(cell: str) -> Cell:
    """One LaTeX cell: \\textbf{...} is the best value, \\textit{...} the second best, -- a missing value, {\\scriptsize$\\pm$s} the standard deviation."""
    cell = cell.strip()
    if cell == "--":
        return Cell(None)
    mark = None
    wrapped = re.fullmatch(r"\\text(bf|it)\{(.*)\}", cell)
    if wrapped:
        mark, cell = ("best" if wrapped.group(1) == "bf" else "second"), wrapped.group(2)
    with_sd = re.fullmatch(r"(.+?)\{\\scriptsize\$\\pm\$([\d.]+)\}", cell)
    return Cell(with_sd.group(1), with_sd.group(2), mark) if with_sd else Cell(cell, None, mark)


def read_table(tex_path: Path) -> tuple[list[str], list[str], list[Row]]:
    """(the settings, the names of the cost columns, the rows) of the tabular in ``table_main_std.tex``."""
    lines = tex_path.read_text().splitlines()
    top = next(l for l in lines if r"\multicolumn" in l)
    groups = re.findall(r"\\multicolumn\{\d+\}\{c\}\{((?:[^{}]|\{[^{}]*\})*)\}", top)  # a group name may hold braces, as in $\pi_{0.5}$
    settings = [g.replace("$\\pi_{0.5}$", "π0.5") for g in groups if not g.startswith("Cost")]
    header = next(i for i, l in enumerate(lines) if l.startswith("Method & Cal."))
    costs = [c.strip().rstrip("\\").strip() for c in lines[header].split("&")][2 + 2 * len(settings) :]
    rows = []
    for line in lines[header + 1 :]:
        if line.startswith(r"\bottomrule"):
            break
        if line.startswith(r"\midrule"):
            continue
        ours = r"\rowcolor" in line
        cells = [c.strip().rstrip("\\").strip() for c in re.sub(r"\\rowcolor\{[^}]*\}\s*", "", line).split("&")]
        rows.append(Row(cells[0].replace("\\_", "_"), ours, cells[1], [unwrap(c) for c in cells[2:]]))
    return settings, costs, rows


def describe(splits: list[str]) -> str:
    """What the runs of a setting are, from the names of its splits: '3 seeds' (paper_seed0, task_seed1, ...), '8 folds' (skill_<name>: one held-out skill
    each) or, on the real robot (loto_<task> for seed 0, lotoseed<k>_<task> for the others), '5 seeds x 8 folds'."""
    if all(s.startswith("skill_") for s in splits):
        return f"{len(splits)} folds"
    if any(s.startswith("loto") for s in splits):
        extra_seeds = {m.group(1) for s in splits if (m := re.match(r"lotoseed(\d+)_", s))}  # seed 0 has no number in its names
        tasks = {s.split("_", 1)[1] for s in splits}
        return f"{1 + len(extra_seeds)} seeds × {len(tasks)} folds"
    seeds = {m.group(1) for s in splits if (m := re.search(r"seed(\d+)", s))}
    return f"{len(seeds)} seeds"


def run_counts(results: Path) -> dict[str, str]:
    """{setting label: what its runs are} for the settings of the paper's table (``fp.eval.figures.SETTINGS``), from the saved runs under ``results``."""
    splits = defaultdict(set)
    for dataset, split, _, _ in iter_runs(results):
        splits[dataset].add(split)
    return {s.label: describe(sorted(n for n in splits[s.dataset] if group_of(n) in (s.group, *s.also))) for s in figures.SETTINGS}


def cell(c: Cell) -> str:
    if c.text is None:
        return "<td>&ndash;</td>"
    sd = f'<span class="sd">&plusmn;{c.sd}</span>' if c.sd else ""
    return f'<td class="{c.mark}">{c.text}{sd}</td>' if c.mark else f"<td>{c.text}{sd}</td>"


def render(settings: list[str], costs: list[str], rows: list[Row], runs: dict[str, str], difficulty: dict[str, str] = DIFFICULTY) -> str:
    """The table as HTML: four header rows (the settings, their runs, their difficulty, seen and unseen) and a row per method (ours shaded; the best value of a column
    bold, the second best italic)."""
    group = "".join(f'<th colspan="2" scope="colgroup">{html.escape(s)}</th>' for s in settings) + f'<th colspan="{len(costs)}" scope="colgroup">Cost (LIBERO-X)</th>'
    counts = "".join(f'<td colspan="2">{html.escape(runs.get(s, ""))}</td>' for s in settings) + f'<td colspan="{len(costs)}"></td>'
    levels = "".join(f'<td colspan="2">{html.escape("difficulty " + difficulty[s]) if s in difficulty else ""}</td>' for s in settings) + f'<td colspan="{len(costs)}"></td>'
    sub = "".join(f'<th scope="col">{e}</th>' for _ in settings for e in ("Seen", "Unseen")) + "".join(f'<th scope="col">{html.escape(c)}</th>' for c in costs)
    body = []
    for r in rows:
        attribute = ' class="ours"' if r.ours else ""
        label = html.escape(OURS if r.ours else r.label)
        body.append(f'<tr{attribute}><th scope="row">{label}</th><td>{r.calibrator}</td>{"".join(cell(c) for c in r.cells)}</tr>')
    return "\n".join([
        '<div class="table-wrap">', '<table class="results-table">', f"<caption>{CAPTION}</caption>", "<thead>",
        f'<tr><th rowspan="4" scope="col">Method</th><th rowspan="4" scope="col">Cal.</th>{group}</tr>', f'<tr class="runs">{counts}</tr>', f'<tr class="difficulty">{levels}</tr>', f"<tr>{sub}</tr>", "</thead>", "<tbody>", *body, "</tbody>", "</table>", "</div>",
    ])


def write_into(page: Path, table: str) -> None:
    """Replace what lies between the markers of the page with the table."""
    text = page.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"{page} must contain the markers {START!r} and {END!r} exactly once")
    page.write_text(pattern.sub(lambda _: f"{START}\n{table}\n{END}", text))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS, help="the saved runs, whose splits are counted")
    p.add_argument("--tex", type=Path, help="the paper's main table with the standard deviations (default: <results>/figures/table_main_std.tex)")
    p.add_argument("--page", type=Path, default=Path("website/index.html"))
    args = p.parse_args()
    settings, costs, rows = read_table(args.tex or args.results / "figures" / "table_main_std.tex")
    runs = run_counts(args.results)
    write_into(args.page, render(settings, costs, rows, runs))
    print(f"wrote the table ({len(rows)} methods, {len(settings)} settings, {len(costs)} cost columns) into {args.page}; runs: {runs}")


if __name__ == "__main__":
    main()
