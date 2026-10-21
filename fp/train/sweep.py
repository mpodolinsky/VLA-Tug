"""Train the probes of a unit (one label variant on one split) and write their runs under results/.

The split's features are loaded and standardised once and shared by the 16 architectures of the unit. Each probe is trained with its
split's seed, then runs on every episode of the dataset; scores, weights, history and (for the architectures that have those heads) the mode and CLIP
head outputs go to ``results/<dataset>/<split>/<method>/``. Runs that
exist are skipped, so a killed sweep resumes; a probe that fails is logged and skipped, never marked finished.
"""

import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .. import config
from ..data.features import load_features
from ..data.manifest import Episode, load_manifest
from ..data.splits import Split, make_splits
from ..data.targets import Targets, load_targets
from ..grid import Unit
from ..models.ours import ProbeConfig, all_configs
from ..results import code_version, is_done, run_dir, split_hash, write_run
from .loop import TrainConfig, default_lr
from .ours import OursScorer
from .standardised import StandardisedFeatures, standardise_features


@dataclass(frozen=True)
class Settings:
    out: Path
    max_epochs: int = 100
    patience: int = 15
    save_weights: bool = True
    save_heads: bool = True  # outputs of the mode and CLIP heads, for the architectures that have them
    min_free_gb: float = 20.0  # below this much free disk, scores and history are still written but weights and head outputs are not


@dataclass(frozen=True)
class Inputs:
    """What a dataset's units need besides the features."""

    dataset: str
    episodes: dict[str, Episode]
    targets: Targets
    splits: dict[str, Split]
    window: int


def load_inputs(dataset: str) -> Inputs:
    episodes = {e.key: e for e in load_manifest(dataset)}
    return Inputs(dataset, episodes, load_targets(dataset), {s.name: s for s in make_splits(dataset)}, config.get(dataset).window_steps)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def head_arrays(scorer: OursScorer, outputs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """The arrays files of the mode and CLIP heads (only the architectures that have them), as ``write_run`` takes them."""
    extras = {}
    if "mode" in outputs:
        extras["mode_logits"] = outputs["mode"]
    if "sem" in outputs:
        extras["semantic"] = outputs["sem"]
        extras["semantic_space"] = {"target": scorer.space.target, "bank_index": scorer.space.bank_index}
    return extras


def train_probe(cfg: ProbeConfig, variant: str, split: Split, inputs: Inputs, features: StandardisedFeatures, settings: Settings,
                device: str) -> tuple[OursScorer, dict, dict]:
    """Fit one probe and run it on every episode; returns the scorer, the outputs of all its heads and the run's history."""
    train_cfg = TrainConfig(lr=default_lr(cfg.trunk), max_epochs=settings.max_epochs, patience=settings.patience, seed=split.seed)
    scorer = OursScorer(cfg, variant, inputs.episodes, features, inputs.targets, inputs.window, train_cfg, device)
    start = time.time()
    scorer.fit(split.train, split.val)
    outputs = scorer.outputs(list(inputs.episodes))
    r = scorer.result
    info = {
        "method": scorer.name, "dataset": inputs.dataset, "split": split.name, "seed": split.seed, "variant": variant, "arch": asdict(cfg),
        "train_cfg": asdict(train_cfg), "window": inputs.window, "best_epoch": r.best_epoch, "epochs": len(r.val["score"]),
        "stopped_early": r.stopped_early, "train": r.train, "val": r.val, "seconds": round(time.time() - start, 1),
        "split_sha256": split_hash(split), "device": device, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return scorer, outputs, info


def train_unit(unit: Unit, inputs: Inputs, settings: Settings, device: str, log: Callable[[str], None] = print,
               architectures: list[str] | None = None) -> int:
    """Train the probes of the unit (all 16 architectures, or only the named ones) that have no run yet; returns how many runs were written."""
    split = inputs.splits[unit.split]
    todo = [cfg for cfg in all_configs() if (architectures is None or cfg.name in architectures)
            and not is_done(run_dir(unit.dataset, unit.split, f"{cfg.name}__{unit.variant}", settings.out))]
    if not todo:
        return 0
    settings.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    features = standardise_features(load_features(unit.dataset, list(inputs.episodes)), split.train, device)
    log(f"{unit.dataset}/{unit.split}/{unit.variant}: features ready in {time.time() - start:.0f}s, {len(todo)} probes to train")
    code = code_version()
    written = 0
    for cfg in todo:
        method = f"{cfg.name}__{unit.variant}"
        try:
            scorer, outputs, info = train_probe(cfg, unit.variant, split, inputs, features, settings, device)
            room = free_gb(settings.out) >= settings.min_free_gb
            weights, heads = settings.save_weights and room, settings.save_heads and room
            write_run(run_dir(unit.dataset, unit.split, method, settings.out), outputs["score"],
                      {**info, "code": code, "weights_saved": weights, "heads_saved": heads}, scorer.save if weights else None,
                      head_arrays(scorer, outputs) if heads else None)
            written += 1
            log(f"  {method}: {info['epochs']} epochs, best {info['best_epoch']}{' (stopped early)' if info['stopped_early'] else ''}, "
                f"val score {info['val']['score'][info['best_epoch']]:.4f}, {info['seconds']:.0f}s{'' if room else ', weights and heads NOT saved (disk)'}")
        except Exception:
            log(f"  FAILED {unit.dataset}/{unit.split}/{method}\n{traceback.format_exc()}")
        finally:
            scorer = None
            torch.cuda.empty_cache()
    return written
