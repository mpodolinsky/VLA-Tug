import numpy as np
import pytest

from fp.calibration.ltt import calibrate_ltt
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval import ltt_compare, tables
from fp.eval.ltt_compare import LTT, MAX_CONFORMAL, compare, ltt_threshold, prepare, strongest_tpr_target

T, ONSET, WINDOW, DELTA = 20, 12, 3, 0.1
STRONGEST = 0.9  # with 60 calibration failures and 40 walks (level 0.1 / 40) a target of 0.9 passes (0.9 ** 60 = 0.0018) and 0.95 does not (0.046)


def world(rises: bool = True):
    """60 calibration successes and failures, 20 of each in the seen and in the unseen set. Every success stays at 0.05 (so the maxima of the
    calibration successes are all 0.05); a failure rises to 0.99 at step 8, before the onset, unless ``rises`` is off."""
    episodes, keys, scores = {}, {part: [] for part in ("train", "val", "calibration", "seen", "unseen")}, {}
    for part, n in (("train", 4), ("val", 4), ("calibration", 60), ("seen", 20), ("unseen", 20)):
        for ok in (True, False):
            for i in range(n):
                key = f"{part}/{'s' if ok else 'f'}{i:02d}"
                episodes[key] = Episode(key, ok, "a", T, None if ok else ONSET)
                keys[part].append(key)
                s = np.full(T, 0.05)
                if not ok and rises:
                    s[8:] = 0.99
                scores[key] = s
    split = Split("task_seed0", keys["train"], keys["val"], keys["calibration"], keys["seen"], keys["unseen"], seed=0)
    return prepare(scores, split, episodes, WINDOW), scores, split, episodes


def test_the_ltt_threshold_is_the_one_calibrate_ltt_gives_on_the_same_episodes():
    run, scores, split, episodes = world()
    cal = [episodes[k] for k in split.calibration]
    fails = [e for e in cal if not e.success]
    for tpr_target, fpr_target in ((0.9, 0.3), (0.5, 0.1)):
        expected = calibrate_ltt([scores[e.key] for e in cal if e.success], [scores[e.key] for e in fails], [e.onset for e in fails],
                                 alpha_tpr_early=1 - tpr_target, alpha_fpr=fpr_target, delta=DELTA, n_starts=40).tau
        assert expected is not None and ltt_threshold(run, tpr_target, fpr_target, DELTA, 40) == expected


def test_the_strongest_tpr_target_is_the_highest_one_that_enough_runs_certify():
    good, _, _, _ = world()
    bad, _, _, _ = world(rises=False)  # failures never rise: nothing is certified
    assert strongest_tpr_target([good, good, bad, bad], 0.3, DELTA, 40) == STRONGEST  # half of the runs are enough
    assert strongest_tpr_target([good, bad, bad, bad], 0.3, DELTA, 40) is None  # a quarter of them are not
    assert strongest_tpr_target([good, bad, bad, bad], 0.3, DELTA, 40, min_share=0.25) == STRONGEST
    assert strongest_tpr_target([good, good, bad], 0.3, DELTA, 40) == STRONGEST  # ceil(0.5 * 3) = 2 runs


def test_the_rows_have_ltt_then_max_conformal_per_fpr_target_and_only_ltt_has_a_tpr_target():
    run, *_ = world()
    rows = compare([run, run], (0.2, 0.3), delta=DELTA)
    assert [(r.fpr_target, r.method) for r in rows] == [(0.2, LTT), (0.2, MAX_CONFORMAL), (0.3, LTT), (0.3, MAX_CONFORMAL)]
    assert [r.highlight for r in rows] == [True, False, True, False]
    ltt, conformal = rows[0], rows[1]
    assert ltt.tpr_target == STRONGEST and conformal.tpr_target is None
    assert (ltt.n_certified, ltt.n_runs, conformal.n_certified, conformal.n_runs) == (2, 2, 2, 2)
    assert ltt.seen.tpr_holds == 100.0 and conformal.seen.tpr_holds is None and conformal.unseen.tpr_holds is None


def test_the_cells_count_the_runs_in_which_each_target_holds_and_take_the_mean_and_spread_of_the_rates():
    """LTT picks a threshold above 0.05, so no success alarms and every failure is caught early. Every calibration success has the maximum 0.05,
    so the max-conformal threshold is 0.05 and (alarm at score >= threshold) every success alarms: the FPR target fails in every run."""
    run, *_ = world()
    ltt, conformal = compare([run, run, run], (0.3,), delta=DELTA)
    assert ltt.seen == tables.LttCells(100.0, 100.0, (0.0, 0.0), (100.0, 0.0))
    assert conformal.seen == tables.LttCells(0.0, None, (100.0, 0.0), (100.0, 0.0))
    assert conformal.unseen == conformal.seen and ltt.unseen == ltt.seen


def test_the_spread_is_over_the_runs():
    """Two runs that differ only in their unseen episodes: in one every failure is caught early, in the other none is."""
    run, scores, split, episodes = world()
    late = {k: (np.full(T, 0.05) if not episodes[k].success and k in split.unseen else s) for k, s in scores.items()}
    other = prepare(late, split, episodes, WINDOW)
    ltt = compare([run, other], (0.3,), delta=DELTA)[0]
    assert ltt.unseen.tpr_early == (50.0, 50.0) and ltt.unseen.tpr_holds == 50.0 and ltt.unseen.fpr == (0.0, 0.0) and ltt.seen.tpr_early == (100.0, 0.0)


def test_a_run_that_cannot_certify_has_no_threshold_and_a_row_with_no_target_has_no_cells():
    bad, *_ = world(rises=False)
    ltt, conformal = compare([bad, bad], (0.3,), delta=DELTA)
    assert (ltt.tpr_target, ltt.n_certified, ltt.seen, ltt.unseen) == (None, 0, None, None)
    assert conformal.n_certified == 2 and conformal.seen is not None  # max-conformal still answers


def test_a_max_conformal_threshold_that_never_alarms_counts_as_no_false_alarm_and_no_detection():
    """alpha below 1 / (n + 1) of 61 maxima: the threshold is +inf."""
    run, *_ = world()
    conformal = compare([run], (0.005,), delta=DELTA)[1]
    assert conformal.seen == tables.LttCells(100.0, None, (0.0, 0.0), (0.0, 0.0))


# ---------------------------------------------------------------- the LaTeX and the CSV
def rows_for_tables():
    run, *_ = world()
    bad, *_ = world(rises=False)
    return compare([run], (0.2, 0.3), delta=DELTA) + compare([bad], (0.4,), delta=DELTA)


def test_the_latex_table_has_both_targets_first_then_the_method_and_shades_only_the_ltt_rows():
    tex = tables.ltt_latex_table(rows_for_tables(), DELTA, 40)
    lines = tex.splitlines()
    assert lines[0].startswith("% python -m cli.ltt_table") and r"\begin{table*}[t]" in tex and r"\label{tab:ltt}" in tex
    assert r"FPR target & TPR-early target & Method & Runs & FPR holds & TPR holds & FPR & TPR early" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex and tex.count(r"\midrule") == 1 + 2  # under the header and between the three FPR blocks
    body = [line for line in lines if line.endswith(r"\\") and ("LTT &" in line or "Max-conformal &" in line)]
    assert len(body) == 6
    for line in body:
        assert line.count("&") == 11  # 12 columns
        assert line.startswith(r"\rowcolor{gray!15} ") == ("LTT &" in line)


def test_the_latex_cells_show_percentages_a_dash_for_a_missing_target_and_the_spread_in_scriptsize():
    lines = tables.ltt_latex_table(rows_for_tables(), DELTA, 40).splitlines()
    ltt = next(line for line in lines if "LTT &" in line)
    conformal = next(line for line in lines if "Max-conformal &" in line)
    abstained = next(line for line in lines if line.startswith(r"\rowcolor{gray!15} 40\%"))
    assert ltt.split(" & ")[:4] == [r"\rowcolor{gray!15} 20\%", r"90\%", "LTT", r"100\% (1/1)"]
    assert conformal.split(" & ")[:4] == ["", "--", "Max-conformal", r"100\% (1/1)"]
    assert r"0\% & -- & 100.0{\scriptsize$\pm$0.0}" in conformal  # FPR holds, no TPR target, FPR 100.0 +- 0.0
    assert abstained.split(" & ")[:4] == [r"\rowcolor{gray!15} 40\%", "--", "LTT", r"0\% (0/1)"] and abstained.endswith(" & ".join(["--"] * 8) + r" \\")


def test_the_caption_states_the_walks_the_confidence_and_the_number_of_runs():
    run, *_ = world()
    caption = tables.ltt_caption(compare([run] * 14, (0.3,), delta=DELTA), 0.1, 40)
    assert "40 fixed-sequence walks" in caption and r"$\delta = 0.1$" in caption and "over 14 runs" in caption and "RUNS" not in caption


def test_the_csv_rows_have_one_record_per_row_with_the_targets_in_percent():
    records = tables.ltt_csv_rows(rows_for_tables())
    assert len(records) == 6
    assert records[0]["fpr target"] == 20.0 and records[0]["tpr-early target"] == 90.0 and records[0]["method"] == LTT
    assert records[1]["tpr-early target"] is None and records[1]["seen tpr holds"] is None
    assert "seen fpr holds" not in records[4] and records[5]["seen fpr mean"] == pytest.approx(100.0)  # the abstaining LTT row has no cells
