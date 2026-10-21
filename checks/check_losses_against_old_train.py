"""The label variants, loss context and loss of fp.train against the earlier experiments' train.py, on the same synthetic data and weights.

    REPO_MAIN=/path/to/project python -I checks/check_losses_against_old_train.py

For each label variant the old make_item / make_context / batch_loss and the new make_item / make_context / probe_loss are run on identical
episodes with identical network weights (the standard lstm_modes_clip recipe); items, context values and the loss must agree.
Not part of the package: it imports the old train.py from the project's scripts/.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OLD = Path(os.environ["REPO_MAIN"]) / "scripts/baselines/ours_libero10"
sys.path.insert(0, str(OLD))
os.environ.setdefault("LIBERO10_POLICY", "groot")

import models as old_models  # noqa: E402  (the old models.py)
import train as old  # noqa: E402  (the old train.py)

from _previous import headnet_to_probe_state  # noqa: E402
from fp.data.manifest import Episode  # noqa: E402
from fp.models.ours import Probe, ProbeConfig  # noqa: E402
from fp.train.items import collate, make_item  # noqa: E402
from fp.train.losses import make_context, probe_loss  # noqa: E402
from fp.train.semantic import fit_semantic_space  # noqa: E402

D, WINDOW, N_PHRASES = 12, old.WINDOW, 100  # at least 64 distinct phrases, so the PCA can keep 64 components (the real data has about 1000)
rng = np.random.default_rng(0)
emb = rng.normal(size=(N_PHRASES, 512)).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

episodes = []  # (Episode, x, phrase ids, mode)
for i in range(30):
    fail = i % 2 == 0
    n = int(rng.integers(15, 30))
    onset = int(rng.integers(3, n - 1)) if fail else None
    episodes.append((Episode(f"e{i}", not fail, "t", n, onset, "a"), rng.normal(size=(n, D)).astype(np.float32),
                     rng.integers(0, N_PHRASES, size=n).astype(np.int32), int(rng.integers(0, 10)) if fail else None))
# The batch comes from the training episodes: the old batch_loss indexes the phrase bank directly, so a phrase outside the training bank
# (possible in a validation batch) crashes it, whereas fp.train.losses leaves such steps out. On training phrases the two are defined alike.
train = episodes[:20]
batch_eps = train[:8]

spec = old_models.SPECS["lstm_modes_clip"]
cfg = ProbeConfig(d_modality=D // 3, modes=True, clip=True)
args = SimpleNamespace(lam=1.0, lam_sem=1.0, lam_rank=1.0)
torch.manual_seed(0)
old_net = old_models.HeadNet(D, "lstm", 10, 64, clip=True)
new_net = Probe(cfg)
new_net.load_state_dict(headnet_to_probe_state(old_net.state_dict()), strict=True)
old_net = old_net.to(old.DEVICE)

for variant in ("onset", "onset_pre", "onset_post", "onset_traj", "onset_pm_cut"):
    def old_item(e, x, ids, mode):
        d = dict(x=x, n=e.n_steps, fail=not e.success, onset=-1 if e.success else e.onset, mode=-1 if e.success else mode, phrase=ids.astype(np.int64))
        return old.make_item(d, variant, torch.from_numpy(x).to(old.DEVICE))

    old_train = [old_item(*e_) for e_ in train]
    old_ctx = old.make_context({"train": old_train}, emb, spec)
    new_train = [make_item(e, torch.from_numpy(x), ids, mode, variant, WINDOW) for e, x, ids, mode in train]
    space = fit_semantic_space(emb, np.concatenate([it.phrase.numpy() for it in new_train]), dim=64)
    new_ctx = make_context(new_train, cfg, space, "cpu")

    items_equal = all(torch.equal(o["y"].cpu(), n.y) and torch.equal(o["x"].cpu(), n.x) and torch.equal(o["phrase"].cpu(), n.phrase)
                      for o, n in zip(old_train, new_train))
    ctx_dev = max(float((old_ctx.mode_weight.cpu() - new_ctx.mode_weight).abs().max()), abs(float(old_ctx.pos_weight) - float(new_ctx.pos_weight)),
                  float((old_ctx.bank.cpu() - new_ctx.bank).abs().max()))
    old_batch = [old_item(*e_) for e_ in batch_eps]
    new_batch = [make_item(e, torch.from_numpy(x), ids, mode, variant, WINDOW) for e, x, ids, mode in batch_eps]
    old_net.train()
    new_net.train()
    a = float(old.batch_loss(old_net, old_batch, spec, old_ctx, args))
    b = float(probe_loss(new_net(collate(new_batch).x), collate(new_batch), new_ctx, cfg))
    print(f"{variant:13s} items equal {items_equal}; context deviation {ctx_dev:.1e}; loss old {a:.6f} new {b:.6f} (difference {abs(a - b):.1e})")
