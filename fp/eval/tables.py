"""The main comparison table as LaTeX and CSV (pure formatting; ``fp.eval.figures`` supplies the numbers).

One row per method, a seen and an unseen column for each setting (the area under TPR early vs FPR up to 40%, in %), a column saying which
calibrator the protocol gives the method (MC max-conformal, FC the functional conformal band of SAFE), and the training cost on LIBERO-X (fit time, state kept to score
a new episode, scoring time per policy step). In every numeric column the best value is bold and the second best italic, judged on the
displayed numbers (so equal displayed values are marked alike); the areas are better when higher, the costs when lower. Our rows are shaded.
The LaTeX needs booktabs and xcolor with the table option (\\rowcolor).
"""

from dataclasses import dataclass

CALIBRATOR_CODES = {"max_conformal": "MC", "safe_band": "FC"}  # MC: max-conformal; FC: functional conformal (the SAFE band)
COST_COLUMNS = (("fit_seconds", "Fit (s)", 1), ("state_mb", "State (MB)", 1), ("score_ms_per_step", "Score (ms/step)", 2))
MISSING = "--"


@dataclass(frozen=True)
class Row:
    label: str
    ours: bool
    calibrator: str  # 'max_conformal' or 'safe_band'
    areas: list[tuple[float, float] | None]  # (mean, std) per setting and set, in the order of ``columns``; None where there is no run
    costs: list[float | None]  # in the order of COST_COLUMNS


def columns(settings: list[str]) -> list[tuple[str, str]]:
    """(setting, set) for every area column: seen then unseen for each setting."""
    return [(s, e) for s in settings for e in ("seen", "unseen")]


def number(value: float, digits: int) -> str:
    """Fixed decimals; two significant digits below 0.1 (a scoring time of 0.0008 ms must not read 0.00); no decimals from 1000 on."""
    if value == 0:
        return f"{0:.{digits}f}"
    if abs(value) < 0.1:
        return f"{value:.2g}"
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"


def marks(values: list[float | None], digits: int, higher_is_better: bool) -> list[str | None]:
    """'best' / 'second' / None for each value, judged on the number as displayed (``number``): every value equal to the best is 'best', every
    one equal to the next distinct value is 'second'."""
    shown = [None if v is None else float(number(v, digits)) for v in values]
    distinct = sorted({s for s in shown if s is not None}, reverse=higher_is_better)
    best, second = (distinct + [None, None])[:2]
    return ["best" if s is not None and s == best else "second" if s is not None and s == second else None for s in shown]


def escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("π0.5", r"$\pi_{0.5}$")


def style(text: str, mark: str | None) -> str:
    return {"best": rf"\textbf{{{text}}}", "second": rf"\textit{{{text}}}"}.get(mark, text)


def tabular(rows: list[Row], settings: list[str], with_std: bool = False) -> str:
    """The bare ``tabular``: header, rows (ours shaded, best and second best marked) and rules."""
    n_area = len(columns(settings))
    area_marks = [marks([r.areas[i][0] if r.areas[i] else None for r in rows], 1, True) for i in range(n_area)]
    cost_marks = [marks([r.costs[i] for r in rows], d, False) for i, (_, _, d) in enumerate(COST_COLUMNS)]
    lines = [r"\begin{tabular}{ll" + "r" * n_area + "r" * len(COST_COLUMNS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{escape(s)}}}" for s in settings) + rf" & \multicolumn{{{len(COST_COLUMNS)}}}{{c}}{{Cost (LIBERO-X)}} \\"]
    lines += [r"".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(settings))) + rf"\cmidrule(lr){{{3 + n_area}-{2 + n_area + len(COST_COLUMNS)}}}"]
    lines += ["Method & Cal. & " + " & ".join(e.capitalize() for _ in settings for e in ("seen", "unseen")) + " & " + " & ".join(name for _, name, _ in COST_COLUMNS) + r" \\",
              r"\midrule"]
    previous_ours = rows[0].ours if rows else False
    for k, r in enumerate(rows):
        if k and r.ours != previous_ours:
            lines.append(r"\midrule")
        previous_ours = r.ours
        cells = []
        for i, area in enumerate(r.areas):
            if area is None:
                cells.append(MISSING)
                continue
            text = number(area[0], 1) + (rf"{{\scriptsize$\pm${area[1]:.1f}}}" if with_std else "")
            cells.append(style(text, area_marks[i][k]))
        cells += [MISSING if c is None else style(number(c, d), cost_marks[i][k]) for i, ((_, _, d), c) in enumerate(zip(COST_COLUMNS, r.costs))]
        prefix = r"\rowcolor{gray!15} " if r.ours else ""
        lines.append(prefix + f"{escape(r.label)} & {CALIBRATOR_CODES[r.calibrator]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


CAPTION = (r"Early failure detection on seen and unseen tasks. Each cell is the area under the curve of TPR early against FPR up to 40\% "
           r"(in \%, higher is better), the mean over the splits of the setting; seen: held-out episodes of tasks the method trained on, "
           r"unseen: held-out tasks. Cal.: calibrator (MC max-conformal, FC functional conformal). Shaded rows are our probe, one architecture "
           r"in every setting, trained with the pre-onset trajectory (\texttt{onset\_pre}) or the whole failed trajectory (\texttt{onset\_traj}) "
           r"as the positive label; SAFE-LSTM and SAFE-MLP use one configuration on the three LIBERO-X splits; every other method uses the "
           r"configuration with the best validation area. Cost, measured on LIBERO-X (lower is better): seconds to fit, megabytes kept to score "
           r"a new episode, milliseconds per policy step. Real robot: GR00T on eight tasks, one task held out at a time, the mean over five seeds "
           r"(40 splits; the configurations are those chosen on seed 0); STAC, ActProbe and FailureSpot need the planned action chunks, which "
           r"were not saved there. Best in bold, second best in italics.")
STD_CAPTION = r" Values after $\pm$ are the standard deviation over the splits."


def latex_table(rows: list[Row], settings: list[str], with_std: bool = False, caption: str | None = None, label: str | None = None,
                source: str = "python -m cli.figures") -> str:
    """The table as a full-width ``table*`` float: the caption above, a label, the tabular scaled to the text width. Needs booktabs, graphicx
    and xcolor with the table option in the preamble."""
    caption = caption or CAPTION + (STD_CAPTION if with_std else "")
    label = label or ("tab:main_std" if with_std else "tab:main")
    return "\n".join([f"% {source}: areas and costs read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      r"\resizebox{\textwidth}{!}{%", tabular(rows, settings, with_std).rstrip("\n") + "%", "}", r"\end{table*}", ""])


def csv_rows(rows: list[Row], settings: list[str]) -> list[dict]:
    """One dict per method: the area mean and std of every setting and set, and the costs."""
    out = []
    for r in rows:
        row = {"method": r.label, "calibrator": r.calibrator}
        for (setting, eval_set), area in zip(columns(settings), r.areas):
            row[f"{setting} {eval_set} mean"], row[f"{setting} {eval_set} std"] = (None, None) if area is None else (round(area[0], 3), round(area[1], 3))
        row.update({name: None if c is None else round(c, 5) for (_, name, _), c in zip(COST_COLUMNS, r.costs)})
        out.append(row)
    return out


# ---------------------------------------------------------------- the failure-mode table
# The mode table has fewer numeric columns than the main table's 15, so scaling it to the full text width would make its text larger than the main
# table's. It is set at a fraction of the text width instead, the main table's scale (it fills the text width) applied to the mode table's own
# natural width, so that both have the same text size. Measured with pdflatex at the normal font size in a two-column article: the natural
# width of the main table is 785.7 pt (10.87 in), that of the mode table 480.6 pt for three settings and 616.4 pt for four, i.e. 72.8 pt for the
# label column plus 135.9 pt per setting: a fraction of 0.093 + 0.173 per setting of the main table's width, 0.61 and 0.78 for three and four
# settings; see docs/FIGURES.md (the sizes need not be identical).
MODE_TABLE_WIDTH_BASE, MODE_TABLE_WIDTH_PER_SETTING = 0.093, 0.173


def mode_table_width(n_settings: int) -> float:
    """The width of the mode table as a fraction of the text width (see above)."""
    return round(MODE_TABLE_WIDTH_BASE + MODE_TABLE_WIDTH_PER_SETTING * n_settings, 2)
MODE_CAPTION = (r"Failure-mode prediction on LIBERO-X (10 modes)REAL: top-1 and top-3 accuracy in \%, the mean over the splits of each kind. "
                r"Most common: always the most common training mode (top-3: the three most common); random: a uniform guess among the training "
                r"modes. Shaded rows are our probe (one architecture), by label variant. The predicted mode is the one with the highest mean "
                r"probability over the WINDOW before an anchor step and the anchor itself: the probe's first alarm (own detection, "
                r"max-conformal at $\alpha = 0.2$, scored on the failures that have an alarm, COVERAGE of them) or the true failure onset (oracle "
                r"onset, an upper bound a deployed system cannot reach, not marked). Best in bold, second best in italics, among the rows other "
                r"than the oracle.")


@dataclass(frozen=True)
class ModeRow:
    label: str
    group: str  # 'baseline', 'detected' (our probe at its own alarm), 'oracle' (at the true onset) or 'oracle_on_detected' (CSV only)
    top1: list[float | None]  # percent, one per setting and set in the order of ``columns``
    top3: list[float | None]
    share: list[float | None] | None = None  # 'detected' rows: percent of the failed episodes that get an alarm, same order
    in_latex: bool = True


REAL_ROBOT = "Real robot"  # the label of the real-robot setting (``fp.eval.figures.SETTINGS``)
REAL_CAPTION = (r" and on the real robot (GR00T on eight tasks, one held out at a time, the mean over five seeds, 40 splits; seven of the ten modes "
                r"occur there, so the random baseline guesses among at most those)")


def mode_caption(rows: list[ModeRow], settings: list[str] | None = None) -> str:
    """``MODE_CAPTION`` with the range of the share of failures that get an alarm read from the rows, and the real robot described when it is one of
    the ``settings`` (the window is 2.4 s on both datasets: six policy steps on LIBERO-X, 20 on the real robot)."""
    shares = [v for r in rows if r.share for v in r.share if v is not None]
    coverage = rf"{min(shares):.0f} to {max(shares):.0f}\%" if shares else "see the CSV"
    real = REAL_ROBOT in (settings or [])
    window = "2.4 s (six steps on LIBERO-X, 20 on the real robot)" if real else "six steps"
    return MODE_CAPTION.replace("COVERAGE", coverage).replace("REAL", REAL_CAPTION if real else "").replace("WINDOW", window)


def mode_latex_table(rows: list[ModeRow], settings: list[str], caption: str | None = None, label: str = "tab:modes",
                     source: str = "python -m cli.figures") -> str:
    """The failure-mode table as a ``table*`` float (see ``latex_table``): top-1 and top-3 side by side, one row per method, the baselines first,
    then our probe at its own alarm and at the oracle onset under their own headings. Only the rows with ``in_latex``."""
    m, n = len(settings), 2 * len(settings)  # settings; numeric columns per metric
    shown = [r for r in rows if r.in_latex]
    compared = [r for r in shown if r.group != "oracle"]  # best and second best are judged among these
    marks_of = {metric: [marks([getattr(r, metric)[i] for r in compared], 1, True) for i in range(n)] for metric in ("top1", "top3")}
    short = [escape(s.removeprefix("LIBERO-X ")) for s in settings]
    lines = [r"\begin{tabular}{l" + "r" * (2 * n) + "}", r"\toprule",
             rf" & \multicolumn{{{n}}}{{c}}{{Top-1 accuracy (\%)}} & \multicolumn{{{n}}}{{c}}{{Top-3 accuracy (\%)}} \\",
             rf"\cmidrule(lr){{2-{1 + n}}}\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for _ in range(2) for s in short) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(2 * m)),
             "Method & " + " & ".join(e.capitalize() for _ in range(2 * m) for e in ("seen", "unseen")) + r" \\"]
    headings = {"detected": "Ours, own detection (first alarm)", "oracle": "Ours, oracle onset (upper bound)"}
    previous = None
    for r in shown:
        if r.group != previous:
            lines.append(r"\midrule")
            if r.group in headings:
                lines.append(rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{{headings[r.group]}}} \\")
        previous = r.group
        index = compared.index(r) if r in compared else None

        def cell(value, metric, i):
            return MISSING if value is None else style(number(value, 1), marks_of[metric][i][index] if index is not None else None)

        cells = [cell(v, "top1", i) for i, v in enumerate(r.top1)] + [cell(v, "top3", i) for i, v in enumerate(r.top3)]
        lines.append(("" if r.group == "baseline" else r"\rowcolor{gray!15} ") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption or mode_caption(rows, settings)}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      rf"\resizebox{{{mode_table_width(m)}\textwidth}}{{!}}{{%", "\n".join(lines) + "%", "}", r"\end{table*}", ""])


def mode_csv_rows(rows: list[ModeRow], settings: list[str]) -> list[dict]:
    """Every row, also those left out of the LaTeX: a column per metric, setting and set."""
    out = []
    for r in rows:
        record = {"method": r.label, "group": r.group}
        for metric, name in (("top1", "top-1"), ("top3", "top-3"), ("share", "alarm share")):
            values = getattr(r, metric)
            if values is not None:
                record.update({f"{name} {s} {e}": None if v is None else round(v, 3) for (s, e), v in zip(columns(settings), values)})
        out.append(record)
    return out


# ---------------------------------------------------------------- the LTT against max-conformal table
LTT_CAPTION = (r"Learn-then-Test (LTT) against max-conformal calibration of the same probe at the same false-positive-rate (FPR) target, over RUNS "
               r"runs (one per split). LTT (STARTS fixed-sequence walks, $\delta = DELTA$) certifies the FPR target together with the TPR-early target "
               r"of the second column, the highest one that at least half of the runs certify; it can abstain, so its percentages are over the runs it "
               r"certifies (column Runs). Max-conformal always returns a threshold and has no TPR-early target. Holds: the share of runs in which the "
               r"target is met on the held-out episodes of the tasks the calibration used (seen) or on held-out tasks (unseen); FPR and TPR early "
               r"(in \%): the mean over the runs and, after $\pm$, the standard deviation. Shaded rows are LTT. The guarantee covers the seen "
               r"episodes only, and the TPR-early targets were chosen from these runs.")


@dataclass(frozen=True)
class LttCells:
    """One evaluation set of one row of the LTT table: the share of runs (%) in which each target holds, and the mean and standard deviation over
    the runs of the FPR and of the TPR early (%)."""
    fpr_holds: float
    tpr_holds: float | None  # None for a method without a TPR-early target
    fpr: tuple[float, float]
    tpr_early: tuple[float, float]


@dataclass(frozen=True)
class LttRow:
    fpr_target: float  # a fraction
    tpr_target: float | None  # a fraction; None for max-conformal, and for LTT when no TPR-early target is certified by enough runs
    method: str
    highlight: bool  # shaded: the method the table argues for
    n_certified: int  # runs with a threshold
    n_runs: int
    seen: LttCells | None  # None when no run has a threshold
    unseen: LttCells | None


def percent(value: float) -> str:
    return rf"{value:.0f}\%"


def ltt_cells(cells: LttCells | None) -> list[str]:
    """FPR holds, TPR holds, FPR, TPR early of one set."""
    if cells is None:
        return [MISSING] * 4
    spread = lambda mean_std: number(mean_std[0], 1) + rf"{{\scriptsize$\pm${mean_std[1]:.1f}}}"
    return [percent(cells.fpr_holds), MISSING if cells.tpr_holds is None else percent(cells.tpr_holds), spread(cells.fpr), spread(cells.tpr_early)]


def ltt_tabular(rows: list[LttRow]) -> str:
    """The bare ``tabular``: both targets first, then the method and its runs, then the seen and the unseen columns; one block per FPR target."""
    lines = [r"\begin{tabular}{lll" + "r" * 9 + "}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{Seen tasks} & \multicolumn{4}{c}{Unseen tasks} \\",
             r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
             r"FPR target & TPR-early target & Method & Runs & " + " & ".join(["FPR holds", "TPR holds", "FPR", "TPR early"] * 2) + r" \\", r"\midrule"]
    previous = None
    for r in rows:
        if r.fpr_target != previous and previous is not None:
            lines.append(r"\midrule")
        first = percent(100 * r.fpr_target) if r.fpr_target != previous else ""
        previous = r.fpr_target
        target = MISSING if r.tpr_target is None else percent(100 * r.tpr_target)
        runs = percent(100 * r.n_certified / r.n_runs) + f" ({r.n_certified}/{r.n_runs})"
        cells = ltt_cells(r.seen) + ltt_cells(r.unseen)
        lines.append((r"\rowcolor{gray!15} " if r.highlight else "") + " & ".join([first, target, escape(r.method), runs] + cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def ltt_caption(rows: list[LttRow], delta: float, n_starts: int) -> str:
    return LTT_CAPTION.replace("RUNS", str(max(r.n_runs for r in rows))).replace("STARTS", str(n_starts)).replace("DELTA", f"{delta:g}")


def ltt_latex_table(rows: list[LttRow], delta: float, n_starts: int, caption: str | None = None, label: str = "tab:ltt",
                    source: str = "python -m cli.ltt_table") -> str:
    """The LTT table as a ``table*`` float (see ``latex_table``). Needs booktabs, graphicx and xcolor with the table option."""
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption or ltt_caption(rows, delta, n_starts)}}}", rf"\label{{{label}}}",
                      r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%", ltt_tabular(rows).rstrip("\n") + "%", "}", r"\end{table*}", ""])


def ltt_csv_rows(rows: list[LttRow]) -> list[dict]:
    """One dict per row: the targets (%), the method, the runs, and for the seen and the unseen set every column of the table."""
    out = []
    for r in rows:
        record = {"fpr target": round(100 * r.fpr_target, 3), "tpr-early target": None if r.tpr_target is None else round(100 * r.tpr_target, 3),
                  "method": r.method, "runs certified": r.n_certified, "runs": r.n_runs}
        for name, cells in (("seen", r.seen), ("unseen", r.unseen)):
            if cells is not None:
                record.update({f"{name} fpr holds": round(cells.fpr_holds, 3), f"{name} tpr holds": None if cells.tpr_holds is None else round(cells.tpr_holds, 3),
                               f"{name} fpr mean": round(cells.fpr[0], 3), f"{name} fpr std": round(cells.fpr[1], 3),
                               f"{name} tpr early mean": round(cells.tpr_early[0], 3), f"{name} tpr early std": round(cells.tpr_early[1], 3)})
        out.append(record)
    return out


# ---------------------------------------------------------------- the ablation table
ABLATION_CAPTION = (r"Ablations of our probe in the simulated settings (LIBERO-10 with $\pi_{0.5}$ and with GR00T, and the three kinds of LIBERO-X held-out set): "
                    r"the area under the curve of TPR early against FPR up to 40\% (in \%, higher is better), the mean "
                    r"over the splits of each setting and, after $\pm$, the standard deviation; in brackets the change relative to our probe "
                    r"(shaded, the same model in both blocks). SPLITS Seen: held-out episodes of tasks the probe trained on; unseen: held-out tasks, "
                    r"signatures or skills. Heads: the architecture (LSTM trunk, no attention) with the pre-onset label, so the rows differ only in the "
                    r"mode head and the CLIP head. Training strategy: the positive label of our architecture: the whole pre-onset trajectory "
                    r"(\texttt{onset\_pre}), every step of the failed rollout (\texttt{onset\_traj}), the window before the onset only (\texttt{onset}), "
                    r"or the window and everything after it (\texttt{onset\_post}). Best in bold, second best in italics, within each block.")
# The table is set at the full text width. Its natural width (863.5 pt with the five settings, measured with pdflatex in a two-column article as for the mode
# table, see above) is larger than the main table's (785.7 pt), so its text is about 9% smaller than the main table's (it would be 559.9 pt, and 0.71 of the
# text width to match, with the three LIBERO-X settings alone).
ABLATION_TABLE_WIDTH = 1.0


@dataclass(frozen=True)
class AblationCell:
    mean: float  # the area, in %
    std: float  # population std over the splits of the setting
    n: int  # number of splits
    change: float | None  # relative change of the mean against the baseline row of the block, in %; None on the baseline row


@dataclass(frozen=True)
class AblationRow:
    label: str
    ours: bool
    cells: list[AblationCell | None]  # one per setting and set in the order of ``columns``


@dataclass(frozen=True)
class AblationBlock:
    title: str
    rows: list[AblationRow]  # the baseline (our probe) first


def signed_percent(value: float) -> str:
    """'+6\\%' or '$-$27\\%', rounded to whole percent; the sign follows the rounded value."""
    shown = round(value)
    return ("+" if shown >= 0 else "$-$") + percent(abs(shown))


def ablation_caption(blocks: list[AblationBlock], settings: list[str]) -> str:
    """``ABLATION_CAPTION`` with the number of splits of each setting read from the first row."""
    counts = [f"{escape(s)} {c.n}" for s, c in zip(settings, blocks[0].rows[0].cells[::2]) if c is not None]
    return ABLATION_CAPTION.replace("SPLITS", f"Number of splits: {', '.join(counts)}.")


def ablation_tabular(blocks: list[AblationBlock], settings: list[str]) -> str:
    """The bare ``tabular``: one block of rows per ablation under its heading, the baseline (shaded) first, best and second best marked per block."""
    m, n = len(settings), 2 * len(settings)
    names = [escape(s) for s in settings]
    lines = [r"\begin{tabular}{l" + "r" * n + "}", r"\toprule", " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in names) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(m)), " & " + " & ".join(e.capitalize() for _ in range(m) for e in ("seen", "unseen")) + r" \\"]
    for block in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{1 + n}}}{{l}}{{{escape(block.title)}}} \\"]
        column_marks = [marks([r.cells[i].mean if r.cells[i] else None for r in block.rows], 1, True) for i in range(n)]
        for k, r in enumerate(block.rows):
            cells = []
            for i, cell in enumerate(r.cells):
                if cell is None:
                    cells.append(MISSING)
                    continue
                text = style(number(cell.mean, 1), column_marks[i][k]) + rf"{{\scriptsize$\pm${cell.std:.1f}}}"
                cells.append(text + ("" if cell.change is None else rf" {{\scriptsize({signed_percent(cell.change)})}}"))
            lines.append((r"\rowcolor{gray!15} " if r.ours else "") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def ablation_latex_table(blocks: list[AblationBlock], settings: list[str], caption: str | None = None, label: str = "tab:ablations",
                         source: str = "python -m cli.figures") -> str:
    """The ablation table as a ``table*`` float (see ``latex_table``). Needs booktabs, graphicx and xcolor with the table option."""
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption or ablation_caption(blocks, settings)}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      rf"\resizebox{{{ABLATION_TABLE_WIDTH}\textwidth}}{{!}}{{%", ablation_tabular(blocks, settings).rstrip("\n") + "%", "}", r"\end{table*}", ""])


def ablation_csv_rows(blocks: list[AblationBlock], settings: list[str]) -> list[dict]:
    """One dict per row: its block and label, and the mean, std and change of every setting and set."""
    out = []
    for block in blocks:
        for r in block.rows:
            record = {"block": block.title, "variant": r.label, "ours": r.ours}
            for (setting, eval_set), cell in zip(columns(settings), r.cells):
                if cell is not None:
                    record.update({f"{setting} {eval_set} mean": round(cell.mean, 3), f"{setting} {eval_set} std": round(cell.std, 3),
                                   f"{setting} {eval_set} change %": None if cell.change is None else round(cell.change, 3)})
            out.append(record)
    return out
