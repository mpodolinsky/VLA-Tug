import csv
import sys

import pytest

from cli import figures as cli_figures
from fp.calibration.protocol import parse_probe_name
from fp.eval import ablations, figures, tables
from fp.eval.ablations import HEADS, VARIANTS, ablation_blocks, make_block, method_name
from fp.models.ours import all_configs

BASE = parse_probe_name(f"{figures.OURS_ARCHITECTURE}__{ablations.BASELINE_VARIANT}")
GROUPS = ("task", "signature", "skill")
SPLITS = {"task": ("task_seed0", "task_seed1"), "signature": ("signature_seed0", "signature_seed1"), "skill": ("skill_a", "skill_b", "skill_c")}


def rows_with(area_of):
    """Evaluated rows of every probe of the suite on two to three splits of each LIBERO-X kind: ``area_of(method, group, eval_set, split_index)``."""
    out = []
    for cfg in all_configs():
        for variant in VARIANTS:
            method = f"{cfg.name}__{variant}"
            for group in GROUPS:
                for i, split in enumerate(SPLITS[group]):
                    for eval_set in ("val", "seen", "unseen"):
                        out.append({"dataset": "liberox_groot", "split": split, "group": group, "method": method, "calibrator": "max_conformal",
                                    "eval_set": eval_set, "area": area_of(method, group, eval_set, i)})
    return out


def flat(offsets):
    """Every split of a probe has the area 50 + offsets.get(method, 0), apart from a spread of +-2 over the splits of a kind."""
    return lambda method, group, eval_set, i: 50.0 + offsets.get(method, 0.0) + (2.0 if i % 2 else -2.0) * (len(SPLITS[group]) > 1)


def test_our_probe_is_the_one_architecture_with_the_pre_onset_label_and_every_row_is_a_probe_of_the_suite():
    assert BASE == {"trunk": "lstm", "mha": False, "heads": "modes_clip", "variant": "onset_pre"}  # the block titles say LSTM, no attention, modes + CLIP, pre-onset
    names = {f"{cfg.name}__{v}" for cfg in all_configs() for v in VARIANTS}
    assert {method_name(BASE, heads=h) for h in HEADS} <= names and {method_name(BASE, variant=v) for v in VARIANTS} <= names
    assert method_name(BASE) == "lstm_modes_clip__onset_pre" and method_name(BASE, heads="plain") == "lstm_plain__onset_pre"
    assert method_name({"trunk": "mlp", "mha": True, "heads": "clip", "variant": "onset"}, heads="modes") == "mlp_mha_modes__onset"
    assert HEADS[0] == "modes_clip" and VARIANTS[0] == ablations.BASELINE_VARIANT  # our probe is the first row of each block


def test_the_blocks_have_our_probe_first_with_no_change_and_the_others_relative_to_it():
    rows = rows_with(flat({"lstm_modes_clip__onset_pre": 10.0, "lstm_plain__onset_pre": 20.0, "lstm_modes_clip__onset_traj": -25.0}))
    heads, strategy = ablation_blocks(rows)
    assert [r.label for r in heads.rows] == ["Modes + CLIP (ours)", "No head", "Modes only", "CLIP only"]
    assert [r.label for r in strategy.rows] == ["onset_pre (ours)", "onset_traj", "onset", "onset_post"]
    assert [r.ours for r in heads.rows] == [True, False, False, False] and [r.ours for r in strategy.rows] == [True, False, False, False]
    assert all(c.change is None for c in heads.rows[0].cells + strategy.rows[0].cells)
    ours, no_head, modes = heads.rows[0], heads.rows[1], heads.rows[2]
    assert len(ours.cells) == 6  # task, signature, skill x seen, unseen
    assert ours.cells[0].mean == pytest.approx(60.0) and no_head.cells[0].mean == pytest.approx(70.0)
    assert no_head.cells[0].change == pytest.approx(100 * 10 / 60) and modes.cells[0].change == pytest.approx(100 * (50 - 60) / 60)
    assert strategy.rows[1].cells[3].change == pytest.approx(100 * (25 - 60) / 60)  # onset_traj, signature unseen


def test_a_cell_is_the_headline_cell_of_the_main_table_mean_population_std_and_number_of_splits():
    rows = rows_with(flat({}))
    cell = ablation_blocks(rows)[0].rows[0].cells
    expected = figures.area_cell(rows, ablations.LIBEROX[2], "lstm_modes_clip__onset_pre", "unseen")
    assert (cell[5].mean, cell[5].std, cell[5].n) == (expected.mean, expected.std, expected.n) and cell[5].n == 3
    assert cell[0].std == pytest.approx(2.0) and [c.n for c in cell[::2]] == [2, 2, 3]  # +-2 over two splits: population std 2


def test_a_probe_without_runs_is_an_error_not_a_gap():
    rows = [r for r in rows_with(flat({})) if r["method"] != "lstm_clip__onset_pre"]
    with pytest.raises(ValueError):
        ablation_blocks(rows)
    assert make_block("x", [("ours", "lstm_modes_clip__onset_pre"), ("none", "lstm_plain__onset_pre")], rows).rows[1].label == "none"  # the other rows are fine


# ---------------------------------------------------------------- the formatting
SETTINGS = [s.label for s in ablations.LIBEROX]


def blocks_for_tables():
    return ablation_blocks(rows_with(flat({"lstm_modes_clip__onset_pre": 10.0, "lstm_plain__onset_pre": 20.0, "lstm_modes_clip__onset_traj": -25.0})))


def test_the_latex_has_a_block_per_ablation_with_booktabs_rules_and_ours_shaded_in_both():
    tex = tables.ablation_latex_table(blocks_for_tables(), SETTINGS)
    lines = tex.splitlines()
    assert lines[0].startswith("% python -m cli.figures") and r"\begin{table*}[t]" in tex and r"\label{tab:ablations}" in tex
    assert r"\begin{tabular}{lrrrrrr}" in tex and r"\multicolumn{2}{c}{task}" in tex and r"\multicolumn{2}{c}{skill}" in tex and "LIBERO-X" not in tex.split(r"\begin{tabular}")[1]
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 1 and tex.count(r"\midrule") == 2 and tex.count(r"\cmidrule") == 3
    assert r"\multicolumn{7}{l}{Heads (LSTM, pre-onset label)} \\" in tex and r"\multicolumn{7}{l}{Training strategy (modes + CLIP heads)} \\" in tex
    shaded = [line for line in lines if line.startswith(r"\rowcolor{gray!15}")]
    assert [line.split(" & ")[0] for line in shaded] == [r"\rowcolor{gray!15} Modes + CLIP (ours)", r"\rowcolor{gray!15} onset\_pre (ours)"]
    body = [line for line in lines if line.endswith(r"\\") and line.count("&") == 6 and "Seen" not in line and "task" not in line]
    assert len(body) == 8  # four rows in each block, seven cells each
    order = [tex.index(s) for s in (r"\begin{table*}", r"\caption{", r"\label{", rf"\resizebox{{{tables.ABLATION_TABLE_WIDTH}\textwidth}}{{!}}{{%", r"\begin{tabular}", r"\end{tabular}%", r"\end{table*}")]
    assert order == sorted(order) and tex.count("{") == tex.count("}")


def test_each_cell_shows_the_area_the_std_and_the_change_but_not_on_our_probe_and_with_a_minus_sign_for_a_drop():
    lines = tables.ablation_latex_table(blocks_for_tables(), SETTINGS).splitlines()
    ours = next(line for line in lines if line.startswith(r"\rowcolor{gray!15} Modes"))
    no_head = next(line for line in lines if line.startswith("No head"))
    onset_traj = next(line for line in lines if line.startswith(r"onset\_traj"))
    assert "(" not in ours.split(" & ", 1)[1] and r"{\scriptsize$\pm$2.0}" in ours  # no change on ours
    assert r"70.0}{\scriptsize$\pm$2.0} {\scriptsize(+17\%)}" in no_head  # +10 on 60: 16.7 -> 17
    assert r"{\scriptsize($-$58\%)}" in onset_traj  # 25 against 60
    assert tables.signed_percent(0.4) == r"+0\%" and tables.signed_percent(-0.4) == r"+0\%" and tables.signed_percent(-6.5) == r"$-$6\%" and tables.signed_percent(6.5) == r"+6\%"


def test_best_and_second_best_are_marked_within_each_block_and_column():
    lines = tables.ablation_latex_table(blocks_for_tables(), SETTINGS).splitlines()
    heads = {line.split(" & ")[0].removeprefix(r"\rowcolor{gray!15} "): line for line in lines if " & " in line}
    # in the heads block "No head" is 20 above ours (best in every column), ours is second best: 60 > 50 (modes, clip)
    assert r"\textbf{" in heads["No head"].split(" & ")[1] and r"\textit{" in heads["Modes + CLIP (ours)"].split(" & ")[1]
    assert "textbf" not in heads["Modes only"] and "textit" not in heads["CLIP only"]
    strategy = [line for line in lines if line.startswith((r"\rowcolor{gray!15} onset\_pre", r"onset\_traj", "onset &", r"onset\_post"))]
    assert all(r"\textbf{" in cell for cell in strategy[0].split(" & ")[1:])  # ours is the best of its block in every column


def test_the_caption_gives_the_splits_of_each_setting_and_describes_the_four_label_variants():
    caption = tables.ablation_caption(blocks_for_tables(), SETTINGS)
    assert "Number of splits: task 2, signature 2, skill 3." in caption and "SPLITS" not in caption
    for variant in (r"\texttt{onset\_pre}", r"\texttt{onset\_traj}", r"\texttt{onset}", r"\texttt{onset\_post}"):
        assert variant in caption


def test_the_csv_has_a_record_per_row_with_mean_std_and_change_for_every_setting_and_set():
    records = tables.ablation_csv_rows(blocks_for_tables(), SETTINGS)
    assert len(records) == 8 and [r["ours"] for r in records] == [True, False, False, False] * 2
    ours, no_head = records[0], records[1]
    assert ours["block"] == "Heads (LSTM, pre-onset label)" and ours["LIBERO-X task seen mean"] == pytest.approx(60.0) and ours["LIBERO-X task seen change %"] is None
    assert no_head["LIBERO-X task seen change %"] == pytest.approx(100 * 10 / 60, abs=1e-3)  # 70 against 60
    assert len([k for k in ours if k.startswith("LIBERO-X")]) == 18  # 3 settings x 2 sets x (mean, std, change)


# ---------------------------------------------------------------- the command
def test_the_command_writes_the_ablation_table_and_says_so_when_a_probe_has_no_runs(tmp_path, capsys):
    args = type("Args", (), {"out": tmp_path})
    cli_figures.write_ablation_table(args, rows_with(flat({})))
    assert (tmp_path / "table_ablations.tex").exists() and "wrote" in capsys.readouterr().out
    records = list(csv.DictReader(open(tmp_path / "table_ablations.csv")))
    assert len(records) == 8 and records[0]["variant"] == "Modes + CLIP (ours)"
    (tmp_path / "table_ablations.tex").unlink()
    cli_figures.write_ablation_table(args, [r for r in rows_with(flat({})) if r["method"] != "lstm_modes__onset_pre"])
    assert "no ablation table" in capsys.readouterr().out and not (tmp_path / "table_ablations.tex").exists()
