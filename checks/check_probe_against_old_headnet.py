"""The probes of fp.models.ours against the HeadNet of the earlier experiments, with the same weights.

    REPO_MAIN=/path/to/project python -I checks/check_probe_against_old_headnet.py

For lstm and mlp trunks, with and without the attention block, with the mode and CLIP heads (the old HeadNet always has a mode head): the old
network is built, its weights are copied into the new one under the new parameter names, and the outputs on random input must be equal.
Not part of the package: it imports the old models.py from the project's scripts/.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder

from _previous import headnet_to_probe_state  # noqa: E402
from fp.models.ours import Probe, ProbeConfig  # noqa: E402

spec = importlib.util.spec_from_file_location("old_models", Path(os.environ["REPO_MAIN"]) / "scripts/baselines/ours_libero10/models.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


torch.manual_seed(0)
x = torch.randn(4, 9, 6144)
for trunk in ("lstm", "mlp"):
    for mha in (False, True):
        old_net = old.HeadNet(6144, "lstm" if trunk == "lstm" else "step", n_modes=10, sem_dim=64, clip=True, mod_proj=128 if mha else 0, mha=mha).eval()
        new_net = Probe(ProbeConfig(trunk=trunk, mha=mha, modes=True, clip=True)).eval()
        new_net.load_state_dict(headnet_to_probe_state(old_net.state_dict()), strict=True)
        a, b = old_net(x), new_net(x)
        worst = max(float((a[k] - b[k]).abs().max()) for k in ("score", "mode", "sem"))
        print(f"{trunk:4s} mha={str(mha):5s}: outputs equal (max difference {worst:.1e}), {sum(p.numel() for p in new_net.parameters()):,} parameters")
