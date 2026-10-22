"""python -m cli.figures [--root results] [--out results/figures] [--costs results/cost_benchmark] [--print-benchmark]

The paper's main table and curve figures, from the evaluated runs under results/ (python -m cli.evaluate first; no model is run):

    table_main.tex / .csv    one row per method: area under TPR early vs FPR (seen and unseen) for each setting, the calibrator, and the cost
                             on LIBERO-X; table_main_std.tex adds the spread over splits; chosen.csv names the configuration behind every cell
    curves_tpr_early         TPR early against FPR for every setting and method (.png and .pdf): a row for the seen tasks on top, one for
                             the unseen tasks below
    curves_tpr_on_time       the same with the on-time TPR

Each method is, in each setting, the configuration with the best mean validation area (one architecture for all of our rows and one LIBERO-X
configuration for each SAFE model; docs/FIGURES.md), with the calibrator the protocol gives it. Our probes carry no per-run costs (they were trained before the sweeps recorded them); --print-benchmark prints the commands that
re-train the probes the table uses into --costs, which is then read for their cost columns.
"""

import argparse
import csv
from pathlib import Path

from fp.calibration.protocol import calibrator_name
from fp.config import RESULTS
from fp.eval import figures, plots, tables
from fp.eval.collect import collect_rows
from fp.eval.compute import collect_costs


def table_rows(rows: list[dict], chosen: dict, costs: dict) -> list[tables.Row]:
    """The rows of the main table: the entries in ``figures.TABLE_LABELS`` that have at least one chosen method."""
    out = []
    for label in figures.TABLE_LABELS:
        entry = next(e for e in figures.ENTRIES if e.label == label)
        methods = [chosen[(s.label, label)] for s in figures.SETTINGS if (s.label, label) in chosen]
        if not methods:
            print(f"no row for {label}: no setting has complete runs")
            continue
        areas = []
        for setting in figures.SETTINGS:
            method = chosen.get((setting.label, label))
            for eval_set in ("seen", "unseen"):
                cell = figures.area_cell(rows, setting, method, eval_set) if method else None
                areas.append(None if cell is None else (cell.mean, cell.std))
        entry_cost = figures.entry_costs(entry, chosen, costs["ours" if entry.kind == "ours" else "baselines"])
        out.append(tables.Row(label, entry.kind == "ours", calibrator_name(methods[0]), areas, [entry_cost[f] for f in figures.COST_FIELDS]))
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    p.add_argument("--costs", type=Path, default=RESULTS / "cost_benchmark", help="runs of our probes re-trained to measure their costs")
    p.add_argument("--print-benchmark", action="store_true", help="print the commands that re-train the probes the table uses, and stop")
    args = p.parse_args()

    rows = collect_rows(args.root)
    chosen, skipped = figures.choose_all(rows)
    for line in skipped:
        print("skipped:", line)

    if args.print_benchmark:
        table_entries = [e for e in figures.ENTRIES if e.label in figures.TABLE_LABELS]
        for arch, variant, splits in figures.benchmark_plan(chosen, rows, table_entries):
            print(f"python -m cli.train --out {args.costs} --datasets liberox_groot --variants {variant} --probes {arch} --splits {' '.join(splits)}")
        return

    costs = {"baselines": collect_costs(args.root, ("liberox_groot",)),
             "ours": collect_costs(args.costs, ("liberox_groot",)) if args.costs.exists() else []}
    settings = [s.label for s in figures.SETTINGS]
    table = table_rows(rows, chosen, costs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_main.tex").write_text(tables.latex_table(table, settings))
    (args.out / "table_main_std.tex").write_text(tables.latex_table(table, settings, with_std=True))
    write_csv(args.out / "table_main.csv", tables.csv_rows(table, settings))
    write_csv(args.out / "chosen.csv", [{"setting": s, "entry": e, "method": m, "calibrator": calibrator_name(m)} for (s, e), m in chosen.items()])
    print("wrote", args.out / "table_main.tex", "(+ _std, .csv) and chosen.csv")

    for metric, ylabel in (("tpr_early", "TPR early (%)"), ("tpr_on_time", "TPR on-time (%)")):
        panels = []
        for eval_set, row_label in (("seen", "Seen tasks"), ("unseen", "Unseen tasks")):  # seen on top
            curves = {(s.label, e.label): figures.mean_curve(rows, s, chosen[(s.label, e.label)], eval_set, metric, args.root)
                      for s in figures.SETTINGS for e in figures.ENTRIES if (s.label, e.label) in chosen}
            panels.append((row_label, curves))
        fig = plots.curves_figure(panels, list(figures.SETTINGS), list(figures.ENTRIES), ylabel)
        print("wrote", *plots.save(fig, f"curves_{metric}", args.out))


if __name__ == "__main__":
    main()
