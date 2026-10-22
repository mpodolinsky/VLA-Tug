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
    return text.replace("_", r"\_").replace("π0.5", r"$\pi_{0.5}$")


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
           r"a new episode, milliseconds per policy step. Best in bold, second best in italics.")
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
