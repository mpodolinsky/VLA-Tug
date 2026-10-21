"""OursScorer: fits one probe (an architecture of ``fp.models.ours`` under one label variant) and scores episodes with it.

fit(train_keys, val_keys):
    standardise the features with the training episodes' statistics -> items under the label variant -> loss context (class weights, and for
    the semantic head the PCA target space of the training phrases) -> train with ``loop.fit`` on the full loss, keep the epoch with the lowest
    validation score term (the weighted BCE alone, so the mode and CLIP heads never choose the epoch). Every term is reported per epoch
    in ``result.train`` / ``result.val``.
score(keys): probability of failure, sigmoid of the score logit, at every step of the full rollout (not cut at the variant's cut: the probe is
causal, so the steps before the cut get the same value either way).
save(path) / load(path): weights, feature statistics and configuration in one file.
"""

from dataclasses import asdict
from typing import Mapping

import numpy as np
import torch

from ..data.features import fit_standardiser, standardise
from ..data.manifest import Episode
from ..data.targets import Targets
from ..models.base import Scorer
from ..models.ours import Probe, ProbeConfig
from .items import Item, collate, make_item
from .loop import FitResult, Step, TrainConfig, default_lr, fit
from .losses import loss_terms, make_context, total_loss
from .semantic import fit_semantic_space


class OursScorer(Scorer):
    def __init__(self, cfg: ProbeConfig, label_variant: str, episodes: Mapping[str, Episode], features: Mapping[str, np.ndarray],
                 targets: Targets, window: int, train_cfg: TrainConfig | None = None, device: str | None = None):
        self.cfg, self.label_variant, self.window = cfg, label_variant, window
        self.episodes, self.features, self.targets = episodes, features, targets
        self.train_cfg = train_cfg or TrainConfig(lr=default_lr(cfg.trunk))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"{cfg.name}__{label_variant}"
        self.model: Probe | None = None
        self.result: FitResult | None = None

    def _item(self, key: str, mean: np.ndarray, std: np.ndarray) -> Item:
        e = self.episodes[key]
        mode = None if e.success else self.targets.modes[key]
        return make_item(e, standardise(self.features[key], mean, std), self.targets.phrase_ids[key], mode, self.label_variant, self.window)

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        self.mean, self.std = fit_standardiser([self.features[k] for k in train_keys])
        train = [self._item(k, self.mean, self.std) for k in train_keys]
        val = [self._item(k, self.mean, self.std) for k in val_keys]
        space = None
        if self.cfg.clip:  # the target space has the width of the semantic head
            space = fit_semantic_space(self.targets.embeddings, np.concatenate([it.phrase.numpy() for it in train]), dim=self.cfg.clip_dim)
        ctx = make_context(train, self.cfg, space, self.device)
        torch.manual_seed(self.train_cfg.seed)
        self.model = Probe(self.cfg).to(self.device)

        def step_fn(model, chunk):
            batch = collate(chunk, self.device)
            terms = loss_terms(model(batch.x), batch, ctx, self.cfg)
            return Step(total_loss(terms, ctx), {name: (float(num), float(den)) for name, (num, den) in terms.items()})

        self.result = fit(self.model, train, val, step_fn, self.train_cfg, select="score")  # early stopping follows the score term only

    @torch.no_grad()
    def score(self, keys: list[str], batch: int = 32) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit() or load() has not been called")
        self.model.eval()
        out = {}
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            x = [standardise(self.features[k], self.mean, self.std) for k in chunk]
            padded = torch.zeros(len(chunk), max(len(a) for a in x), self.cfg.d_in)
            for j, a in enumerate(x):
                padded[j, : len(a)] = torch.from_numpy(a)
            prob = torch.sigmoid(self.model(padded.to(self.device))["score"]).cpu().numpy().astype(np.float32)
            out.update({k: prob[j, : len(x[j])] for j, k in enumerate(chunk)})
        return out

    def save(self, path: str) -> None:
        torch.save({"cfg": asdict(self.cfg), "variant": self.label_variant, "state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "best_epoch": self.result.best_epoch if self.result else None}, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["cfg"] != asdict(self.cfg) or saved["variant"] != self.label_variant:
            raise ValueError(f"{path} holds {saved['cfg']} / {saved['variant']}, this scorer is {asdict(self.cfg)} / {self.label_variant}")
        self.model = Probe(self.cfg).to(self.device)
        self.model.load_state_dict(saved["state"])
        self.mean, self.std = saved["mean"], saved["std"]
