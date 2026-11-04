"""The pure logic of site_export: the draw of the episodes, the alarm records, the Gaussians and the paths (synthetic data; no results needed)."""

import numpy as np
import pytest

from fp.data.manifest import Episode
from fp.data.splits import Split
from site_export import embeddings, episodes, videos
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
