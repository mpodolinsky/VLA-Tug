"""Which episodes the figure shows (a random draw from the unseen ones, or episodes chosen by hand), and everything the page needs about one of them (as plain lists, ready for JSON)."""

import random
from typing import Mapping

import numpy as np

from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.data.targets import MODE_NAMES
from fp.metrics.outcomes import alarm_step, classify

from .calibrate import Alarm


def select(split: Split, episodes: Mapping[str, Episode], seed: int, n_successes: int, n_failures: int) -> list[str]:
    """A random draw from the split's unseen episodes (the held-out tasks: no probe or calibrator has seen them), with a fixed seed:
    ``n_successes`` successes, then ``n_failures`` failures."""
    rng = random.Random(seed)
    unseen = sorted(split.unseen)
    successes = [k for k in unseen if episodes[k].success]
    failures = [k for k in unseen if not episodes[k].success]
    return rng.sample(successes, n_successes) + rng.sample(failures, n_failures)


def chosen(split: Split, keys: list[str]) -> list[str]:
    """Episodes picked by hand: each must be an unseen episode of the split (no probe or calibrator has seen its task), and none may be repeated."""
    if len(set(keys)) != len(keys):
        raise ValueError("an episode is listed twice")
    not_unseen = [k for k in keys if k not in split.unseen]
    if not_unseen:
        raise ValueError(f"not unseen episodes of this split: {not_unseen}")
    return list(keys)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def rounded(a: np.ndarray, digits: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


def method_record(alarm: Alarm, scores: np.ndarray, onset: int | None, window: int) -> dict:
    """One method on one episode: its scores, its threshold, its alarm step and how the alarm compares with the failure time."""
    threshold = alarm.for_episode(len(scores))
    return {
        "label": alarm.label,
        "scores": rounded(scores, 4),
        "threshold": None if alarm.threshold is None else (round(float(threshold[0]), 4) if alarm.constant else rounded(threshold, 4)),
        "alarm_step": alarm_step(scores, threshold),
        "outcome": classify(scores, threshold, onset, window).value,
    }


def episode_record(eid: str, key: str, episode: Episode, instruction: str, mode: int | None, step_seconds: float, window: int, alarms: Mapping[str, Alarm],
                   scores: Mapping[str, Mapping[str, np.ndarray]], mode_logits: np.ndarray, paths: Mapping[str, np.ndarray], video: Mapping) -> dict:
    """The page's data for one episode. Every per-step list has ``episode.n_steps`` entries."""
    return {
        "id": eid,
        "key": key,
        "task": episode.task,
        "instruction": instruction,
        "success": episode.success,
        "mode": None if mode is None else MODE_NAMES[mode],
        "onset_step": episode.onset,
        "n_steps": episode.n_steps,
        "step_seconds": step_seconds,
        "window_steps": window,
        "video": dict(video),
        "methods": {name: method_record(alarm, scores[name][key], episode.onset, window) for name, alarm in alarms.items()},
        "mode_probs": rounded(softmax(mode_logits), 4),
        "embeddings": {layer: rounded(path, 3) for layer, path in paths.items()},
    }
