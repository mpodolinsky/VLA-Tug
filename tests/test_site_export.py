"""The pure logic of site_export: the draw of the episodes, the alarm records, the Gaussians and the paths (synthetic data; no results needed)."""

import numpy as np
import pytest

from fp.data.manifest import Episode
from fp.data.splits import Split
from site_export import embeddings, episodes, export, main_table, videos
from site_export.calibrate import Alarm


def make_split():
    eps = {}
    for i in range(10):
        eps[f"s{i}"] = Episode(f"s{i}", True, "task", 20, None)
    for i in range(10):
        eps[f"f{i}"] = Episode(f"f{i}", False, "task", 20, 8)
    train, calibration = ["s0", "f0"], ["s1", "f1"]
    unseen = [k for k in eps if k not in train + calibration]
    return Split("task_seed0", train, [], calibration, [], unseen, 0), eps


def test_draw_takes_only_unseen_episodes_with_the_requested_outcomes():
    split, eps = make_split()
    keys = episodes.select(split, eps, seed=0, n_successes=1, n_failures=2)
    assert len(keys) == 3 and set(keys) <= set(split.unseen)
    assert [eps[k].success for k in keys] == [True, False, False]


def test_draw_is_fixed_by_the_seed():
    split, eps = make_split()
    assert episodes.select(split, eps, 3, 1, 2) == episodes.select(split, eps, 3, 1, 2)
    assert any(episodes.select(split, eps, s, 1, 2) != episodes.select(split, eps, 0, 1, 2) for s in range(1, 10))


def test_chosen_episodes_must_be_unseen_and_distinct_and_keep_their_order():
    split, _ = make_split()
    keys = [split.unseen[3], split.unseen[1]]
    assert episodes.chosen(split, keys) == keys
    with pytest.raises(ValueError):
        episodes.chosen(split, ["s0"])  # a training episode
    with pytest.raises(ValueError):
        episodes.chosen(split, [keys[0], keys[0]])


def test_the_content_version_changes_when_a_video_or_a_data_file_changes_and_not_otherwise(tmp_path):
    (tmp_path / "data" / "episodes").mkdir(parents=True)
    (tmp_path / "data" / "embeddings").mkdir()
    (tmp_path / "media" / "rollouts" / "ep1").mkdir(parents=True)
    (tmp_path / "data" / "episodes" / "ep1.json").write_text("{}")
    (tmp_path / "media" / "rollouts" / "ep1" / "agent.mp4").write_bytes(b"old video")
    first = export.content_version(tmp_path)
    assert first == export.content_version(tmp_path) and len(first) == 10
    (tmp_path / "media" / "rollouts" / "ep1" / "agent.mp4").write_bytes(b"new video")  # same file name, new content
    assert export.content_version(tmp_path) != first


def test_the_signature_table_maps_level_and_task_to_the_signature_and_whether_it_is_held_out(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("sig,level,file,split_leave_signature_out\n"
                    "toggle_appliance_off | stove | - | stove | single,LEVEL3,EXTENSION_KITCHEN_SCENE32_LEVEL3__T186_turn_off_the_flat_stove.bddl,test\n"
                    "place_in_container | bowl | in/inside | basket | single,LEVEL1,EXTENSION_KITCHEN_SCENE5_place_the_red_bowl_in_the_basket.bddl,train\n")
    table = export.signature_table(path)
    assert table[("LEVEL3", "EXTENSION_KITCHEN_SCENE32_LEVEL3__T186_turn_off_the_flat_stove")] == ("toggle_appliance_off | stove | - | stove | single", True)
    assert table[("LEVEL1", "EXTENSION_KITCHEN_SCENE5_place_the_red_bowl_in_the_basket")][1] is False
    assert export.signature_table(tmp_path / "missing.csv") == {}  # LIBERO-10 has no signatures


def test_callouts_are_sorted_by_time_and_must_lie_inside_the_video(tmp_path):
    path = tmp_path / "callouts.json"
    path.write_text('{"ep": [{"time": 9.5, "text": " second "}, {"time": 4.4, "text": "first"}]}')
    assert export.callouts_for(path, "ep", 16.8) == [{"time": 4.4, "text": "first"}, {"time": 9.5, "text": "second"}]
    assert export.callouts_for(path, "other episode", 16.8) == [] and export.callouts_for(tmp_path / "missing.json", "ep", 16.8) == []
    with pytest.raises(ValueError):
        export.callouts_for(path, "ep", 9.0)  # a note after the end of the video


def test_novelty_names_what_is_unseen_in_the_episodes_split():
    sig = "toggle_appliance_off | stove | - | stove | single"
    assert export.novelty("signature_seed2", sig, True) == {"label": "Novel task structure (unseen signature): ", "value": sig}
    assert export.novelty("signature_seed2", sig, False)["label"] == "Task structure (signature): "  # a signature that was trained on is not novel
    assert export.novelty("skill_place_on_surface", "place_on_surface | bowl | on top of | plate | compound", False) == {"label": "Novel skill (unseen): ", "value": "place_on_surface"}
    assert export.novelty("task_seed0", sig, False)["label"] == "Task structure (signature): " and export.novelty("task_seed0", None, False) is None


def test_the_split_is_described_for_the_page():
    assert export.split_description("skill_place_on_surface") == "a new skill: place on surface"
    assert export.split_description("signature_seed1").startswith("new task signatures") and export.split_description("task_seed2") == "new tasks"


def test_groups_come_from_a_set_file_with_their_own_operating_points_or_from_the_command_line(tmp_path):
    import argparse

    config = tmp_path / "set.json"
    config.write_text('{"criteria": "why", "groups": [{"split": "a", "keys": ["k1"], "fpr": 0.3, "tpr_early": 0.45}, {"split": "b", "keys": ["k2"]}]}')
    parser = argparse.ArgumentParser()
    groups, criteria = export.load_groups(argparse.Namespace(set=config), parser)
    assert criteria == "why" and groups[0]["fpr"] == 0.3 and groups[1]["fpr"] == export.FPR_TARGET and groups[1]["tpr_early"] == export.TPR_EARLY_TARGET
    config.write_text('{"groups": [{"split": "a"}]}')
    with pytest.raises(SystemExit):
        export.load_groups(argparse.Namespace(set=config), parser)  # a group needs keys
    one, text = export.load_groups(argparse.Namespace(set=None, split="s", fpr=0.2, tpr_early=0.4, keys=["k"], criteria="c"), parser)
    assert one == [{"split": "s", "fpr": 0.2, "tpr_early": 0.4, "keys": ["k"]}] and text == "c"


def test_softmax_rows_sum_to_one_and_keep_the_order():
    p = episodes.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1) and p[0].argmax() == 2 and np.allclose(p[1], 1 / 3)


def test_constant_threshold_is_exported_as_a_number_and_a_varying_one_as_a_list():
    scores = np.array([0.1, 0.5, 0.9, 0.9])
    constant = Alarm("A", "m", "text", np.full(10, 0.8))
    varying = Alarm("B", "m", "text", np.linspace(0.2, 1.0, 10))
    a = episodes.method_record(constant, scores, onset=3, window=1)
    b = episodes.method_record(varying, scores, onset=3, window=1)
    assert a["threshold"] == 0.8 and a["alarm_step"] == 2 and a["outcome"] == "on_time"
    assert isinstance(b["threshold"], list) and len(b["threshold"]) == 4


def test_a_method_without_a_certified_threshold_never_alarms():
    record = episodes.method_record(Alarm("A", "m", "text", None), np.array([0.99, 0.99]), onset=1, window=1)
    assert record["alarm_step"] is None and record["threshold"] is None and record["outcome"] == "missed"
    assert episodes.method_record(Alarm("A", "m", "text", None), np.array([0.99]), onset=None, window=1)["outcome"] == "correct_reject"


def test_gaussian_axes_reproduce_the_covariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3)) @ np.diag([3.0, 1.0, 0.5])
    mean, axes = embeddings.gaussian(x.mean(axis=0), np.cov(x.T))
    assert np.allclose(np.array(axes).T @ np.array(axes), np.cov(x.T), atol=2e-3) and np.allclose(mean, x.mean(axis=0), atol=1e-3)


def test_episode_path_is_ordered_by_step():
    points = {"keys": np.array(["a", "b"]), "episode": np.array([1, 0, 1, 1, 0]), "step": np.array([1, 0, 0, 2, 1]),
              "xyz": np.arange(15, dtype=np.float32).reshape(5, 3)}
    assert embeddings.episode_path(points, "b")[:, 0].tolist() == [6.0, 0.0, 9.0]  # steps 0, 1, 2 of episode 'b' (rows 2, 0, 3)
    assert embeddings.episode_path(points, "a")[:, 0].tolist() == [3.0, 12.0]


def test_only_h264_yuv420p_videos_are_copied_unchanged():
    assert videos.browser_ready({"codec": "h264", "pix_fmt": "yuv420p"})
    assert not videos.browser_ready({"codec": "mpeg4", "pix_fmt": "yuv420p"})
    assert not videos.browser_ready({"codec": "h264", "pix_fmt": "yuv444p"})


TEX = r"""\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{LIBERO-10 $\pi_{0.5}$} & \multicolumn{2}{c}{Real robot} & \multicolumn{2}{c}{Cost (LIBERO-X)} \\
Method & Cal. & Seen & Unseen & Seen & Unseen & Fit (s) & GPU (ms) \\
\midrule
SAFE-LSTM & FC & \textbf{75.5{\scriptsize$\pm$13.4}} & 34.9{\scriptsize$\pm$3.5} & \textit{44.5{\scriptsize$\pm$13.5}} & 25.1{\scriptsize$\pm$18.8} & 43.5 & 0.066 \\
STAC & MC & 33.8{\scriptsize$\pm$5.1} & 36.4{\scriptsize$\pm$5.3} & -- & -- & \textbf{0.0} & 21.30 \\
\midrule
\rowcolor{gray!15} Ours (onset\_pre) & MC & \textit{75.1{\scriptsize$\pm$12.5}} & \textbf{47.9{\scriptsize$\pm$20.9}} & \textbf{47.6{\scriptsize$\pm$10.9}} & \textbf{25.8{\scriptsize$\pm$9.0}} & 6.8 & 0.12 \\
\bottomrule
\end{tabular}"""
RUNS = {"LIBERO-10 π0.5": "3 seeds", "Real robot": "5 seeds × 8 folds"}
LEVELS = {"LIBERO-10 π0.5": "*"}


def test_the_table_of_the_paper_is_read_with_its_marks_deviations_and_dashes(tmp_path):
    path = tmp_path / "table_main_std.tex"
    path.write_text(TEX)
    settings, costs, rows = main_table.read_table(path)
    assert settings == ["LIBERO-10 π0.5", "Real robot"] and costs == ["Fit (s)", "GPU (ms)"]
    assert [r.label for r in rows] == ["SAFE-LSTM", "STAC", "Ours (onset_pre)"] and [r.ours for r in rows] == [False, False, True]
    assert rows[0].cells[0] == main_table.Cell("75.5", "13.4", "best") and rows[0].cells[2] == main_table.Cell("44.5", "13.5", "second")
    assert rows[0].cells[1] == main_table.Cell("34.9", "3.5", None) and rows[0].cells[4] == main_table.Cell("43.5", None, None)
    assert rows[1].cells[2] == main_table.Cell(None) and rows[1].cells[4] == main_table.Cell("0.0", None, "best")


def test_the_html_table_keeps_values_marks_and_deviations_and_lists_the_runs(tmp_path):
    path = tmp_path / "table_main_std.tex"
    path.write_text(TEX)
    out = main_table.render(*main_table.read_table(path), RUNS, LEVELS)
    assert out.count("<tr") == 4 + 3 and '<tr class="ours"><th scope="row">VLA-Tug (ours)</th>' in out
    assert '<td class="best">75.5<span class="sd">&plusmn;13.4</span></td>' in out and '<td class="second">44.5<span class="sd">&plusmn;13.5</span></td>' in out
    assert "<td>34.9<span class=\"sd\">&plusmn;3.5</span></td>" in out and "<td>&ndash;</td>" in out and '<td class="best">0.0</td>' in out
    assert '<tr class="runs"><td colspan="2">3 seeds</td><td colspan="2">5 seeds × 8 folds</td><td colspan="2"></td></tr>' in out
    assert '<tr class="difficulty"><td colspan="2">difficulty *</td><td colspan="2"></td><td colspan="2"></td></tr>' in out and 'rowspan="4"' in out
    assert "Ours (onset_pre)" not in out and main_table.CAPTION in out


def test_what_the_runs_of_a_setting_are_comes_from_the_names_of_its_splits():
    assert main_table.describe(["paper_seed0", "paper_seed1", "paper_seed2"]) == "3 seeds"
    assert main_table.describe(["task_seed0", "task_seed2"]) == "2 seeds"
    assert main_table.describe(["skill_articulate_open", "skill_stack_objects", "skill_place_on_surface"]) == "3 folds"
    real = [f"loto_{t}" for t in ("basket", "pineapple_out_of_drawer")] + [f"lotoseed{k}_{t}" for k in (1, 2, 3, 4) for t in ("basket", "pineapple_out_of_drawer")]
    assert main_table.describe(real) == "5 seeds × 2 folds"


def test_the_table_is_written_between_the_markers_and_rewritten_in_place(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(f"<main>\n{main_table.START}\nold\n{main_table.END}\n</main>")
    main_table.write_into(page, "<table>one</table>")
    main_table.write_into(page, "<table>two</table>")
    assert page.read_text() == f"<main>\n{main_table.START}\n<table>two</table>\n{main_table.END}\n</main>"
    page.write_text("<main></main>")
    with pytest.raises(ValueError):
        main_table.write_into(page, "<table></table>")
