"""The ablations of our probe on LIBERO-X, from the evaluated runs (the table of ``cli.figures``; no model is run).

Our probe is one architecture (``figures.OURS_ARCHITECTURE``: LSTM trunk, no attention, mode and CLIP heads) trained with the pre-onset label. Two things
are varied one at a time from it, and every row is a probe that was trained on every split:

    heads              the same architecture and label with no head, with the mode head only, with the CLIP head only (and with both: ours)
    training strategy  the same architecture with each of the four label variants (``fp.data.labels``)

Each cell is the headline cell of the main table (``figures.area_cell``): the area under TPR early vs FPR up to 40%, with the calibrator the protocol gives
our probes (max-conformal), the mean over the splits of a kind of held-out set (task, signature, skill) and the population standard deviation, for the seen
and the unseen episodes. Besides, the relative change of the mean against our probe, the first row of each block.
"""

from ..calibration.protocol import parse_probe_name
from . import figures
from .modes import LIBEROX
from .tables import AblationBlock, AblationCell, AblationRow

BASELINE_VARIANT = "onset_pre"  # the label of our probe
HEADS = ("modes_clip", "plain", "modes", "clip")  # ours first
HEAD_LABELS = {"modes_clip": "Modes + CLIP (ours)", "plain": "No head", "modes": "Modes only", "clip": "CLIP only"}
VARIANTS = ("onset_pre", "onset_traj", "onset", "onset_post")  # ours first
SETS = ("seen", "unseen")


def method_name(base: dict, heads: str | None = None, variant: str | None = None) -> str:
    """The probe ``base`` (the factors of ``parse_probe_name``) with its heads or its label variant replaced."""
    return f"{base['trunk']}{'_mha' if base['mha'] else ''}_{heads or base['heads']}__{variant or base['variant']}"


def make_block(title: str, entries: list[tuple[str, str]], rows: list[dict], settings=LIBEROX) -> AblationBlock:
    """The block of ``entries`` (label, method), the first of which is the baseline the changes are relative to. Raises ``ValueError`` when a method has
    no area in a setting (``figures.area_cell``)."""
    cells = {label: [figures.area_cell(rows, s, method, e) for s in settings for e in SETS] for label, method in entries}
    baseline = cells[entries[0][0]]
    return AblationBlock(title, [
        AblationRow(label, k == 0, [AblationCell(c.mean, c.std, c.n, None if k == 0 else 100 * (c.mean - b.mean) / b.mean) for c, b in zip(cells[label], baseline)])
        for k, (label, _) in enumerate(entries)])


def ablation_blocks(rows: list[dict], settings=LIBEROX) -> list[AblationBlock]:
    """The two blocks, heads and training strategy, each with our probe first."""
    base = parse_probe_name(f"{figures.OURS_ARCHITECTURE}__{BASELINE_VARIANT}")
    return [make_block("Heads (LSTM, pre-onset label)", [(HEAD_LABELS[h], method_name(base, heads=h)) for h in HEADS], rows, settings),
            make_block("Training strategy (modes + CLIP heads)", [(v + (" (ours)" if v == BASELINE_VARIANT else ""), method_name(base, variant=v)) for v in VARIANTS], rows, settings)]
