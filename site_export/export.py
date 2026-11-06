"""Export the data of the project page's interactive figure.

    python -m site_export.export --list                       # which episodes are drawn, and the command that fetches their videos
    python -m site_export.export --split <split> --keys <episode keys> --criteria "<one sentence>" --videos <folder>   # chosen episodes instead of the random draw
    python -m site_export.export --videos <folder>            # write website/data and website/media

Run it from the root of a checkout that has ``data/`` and ``results/`` (README.md of this folder). It reads the saved runs of one split: the scores of the
three methods, the calibrators fitted on the split's calibration episodes (``calibrate.py``), the mode head's output and the saved principal-axis points of
three layers of the probe (``embeddings.py``), and writes plain JSON and the videos (``videos.py``) for the page. Nothing is trained.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from fp.config import RESULTS, get
from fp.data.inputs import load_inputs
from fp.data.targets import MODE_NAMES
from fp.eval import figures
from fp.eval.collect import collect_rows
from fp.eval.run_eval import rates_at
from fp.eval.select_config import group_of
from fp.eval.tsne import load_points
from fp.results import read_scores, run_dir

from . import embeddings, episodes, videos
from .calibrate import FPR_TARGET, TPR_EARLY_TARGET, Alarm, calibrate

# what the unseen episodes of a split are, as the page says it (the group of the split name: task, signature or skill)
UNSEEN_KIND = {"task": "new tasks", "signature": "new task signatures: familiar components in a new combination", "skill": "a skill that is new"}

# the three methods of the figure: the page's key, the entry of the paper's main table that names the configuration, and the calibrator
METHODS = (("ours", "Ours (onset_pre)", "ltt", "VLA-Tug"), ("safe_lstm", "SAFE-LSTM", "safe_band", "SAFE-LSTM"), ("stac", "STAC", "max_conformal", "STAC"))


def write_json(path: Path, obj, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, separators=None if indent else (",", ":")))


def content_version(out: Path) -> str:
    """A short hash of every exported episode, layer and video: the page puts it on the URL of each file, so a browser that has cached an earlier export (with the same
    file names) fetches the new files."""
    digest = hashlib.sha1()
    for folder in ("data/episodes", "data/embeddings", "media/rollouts"):
        for path in sorted((out / folder).rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(out)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def instruction_of(dataset: str, key: str) -> dict:
    with open(get(dataset).data_dir / "manifest.csv", newline="") as f:
        return next(row for row in csv.DictReader(f) if row["dir"] == key)


def signature_table(path: Path) -> dict[tuple[str, str], tuple[str, bool]]:
    """{(level, task): (signature, held out in the leave-signature-out split)} from the dataset's splits/splits.csv (LIBERO-X); empty if the file does not exist.
    A signature is the combination of a task's skill, target class, relation, landmark and whether it is a compound task."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["level"], r["file"].removesuffix(".bddl")): (r["sig"], r["split_leave_signature_out"] == "test") for r in csv.DictReader(f)}


def callouts_for(path: Path, key: str, duration: float) -> list[dict]:
    """The numbered notes the page draws on the score timeline of an episode, from ``callouts.json`` ({episode key: [{"time": seconds, "text": ...}]}),
    sorted by time; none if the file or the episode is missing. A time outside the video or an empty text is an error."""
    if not path.exists():
        return []
    notes = json.loads(path.read_text()).get(key, [])
    for note in notes:
        if not 0 <= note["time"] <= duration or not note["text"].strip():
            raise ValueError(f"{key}: bad callout {note}")
    return sorted(({"time": n["time"], "text": n["text"].strip()} for n in notes), key=lambda n: n["time"])


def realised_rates(alarms: dict[str, Alarm], scores: dict, split, inputs) -> dict[str, dict]:
    """What each threshold gives on the calibration episodes and on all unseen ones (``fp.eval.run_eval.rates_at``): FPR and TPR early in percent."""
    rates = {}
    for name, alarm in alarms.items():
        if alarm.threshold is None:
            continue
        threshold = alarm.threshold[0] if alarm.constant else (lambda n, t=alarm.threshold: t[:n])
        rates[name] = {title: {k: round(r[k], 1) for k in ("fpr", "tpr_early")} | {"episodes": len(keys)}
                       for title, keys in (("calibration", split.calibration), ("unseen", split.unseen))
                       for r in [rates_at(threshold, scores[name], [inputs.episodes[k] for k in keys], inputs.window)]}
    return rates


def check_report(alarms: dict[str, Alarm], rates: dict[str, dict]) -> list[str]:
    lines = []
    for name, alarm in alarms.items():
        if name not in rates:
            lines.append(f"{alarm.label:10s} not certified: no threshold")
            continue
        shown = f"{alarm.threshold[0]:.4f}" if alarm.constant else f"{alarm.threshold[0]:.3f} .. {alarm.threshold[-1]:.3f} (varies with the step)"
        parts = [f"{title}: FPR {r['fpr']}% TPR early {r['tpr_early']}%" for title, r in rates[name].items()]
        lines.append(f"{alarm.label:10s} threshold {shown:34s} " + " | ".join(parts))
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="liberox_groot")
    p.add_argument("--split", default="task_seed0", help="a split of the dataset; the episodes drawn are its unseen ones")
    p.add_argument("--seed", type=int, default=0, help="the seed of the random draw of the episodes")
    p.add_argument("--successes", type=int, default=1)
    p.add_argument("--failures", type=int, default=2)
    p.add_argument("--keys", nargs="+", help="show these episodes, in this order, instead of the random draw; each must be an unseen episode of the split")
    p.add_argument("--criteria", default="", help="with --keys: one sentence on how the episodes were chosen; the page shows it under the figure")
    p.add_argument("--callouts", type=Path, default=Path(__file__).with_name("callouts.json"), help="numbered notes on the score timeline of chosen episodes (see callouts.json)")
    p.add_argument("--videos", type=Path, help="a folder with <episode key>/rollout.mp4 and wrist.mp4 of the chosen episodes (see --list)")
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=Path("website"), help="the folder of the page: data/ and media/ are written into it")
    p.add_argument("--list", action="store_true", help="only print the chosen episodes")
    args = p.parse_args()

    inputs = load_inputs(args.dataset)
    split = inputs.splits[args.split]
    keys = episodes.chosen(split, args.keys) if args.keys else episodes.select(split, inputs.episodes, args.seed, args.successes, args.failures)
    spec = get(args.dataset)
    if args.list:
        print(f"{len(keys)} chosen unseen episodes of {args.split}:" if args.keys else
              f"{args.successes} success(es) and {args.failures} failure(s), drawn with seed {args.seed} from the {len(split.unseen)} unseen episodes of {args.split}:")
        print(*keys, sep="\n")
        print(f"\nfetch their videos (they are in the released dataset {spec.hf_repo}, which needs an approved access request):\n"
              f"  hf download {spec.hf_repo} --repo-type dataset --local-dir <folder> " + " ".join(f'--include "{k}/*.mp4"' for k in keys))
        return
    if args.videos is None:
        p.error("--videos is required (use --list to see which episodes and how to get their videos)")

    # which configuration each method stands for: the one of the paper's main table in the split's setting
    setting = next(s for s in figures.SETTINGS if s.dataset == args.dataset and s.group == group_of(args.split))
    entries = {e.label: e for e in figures.ENTRIES}
    rows = collect_rows(args.results, (args.dataset,))
    methods = {name: figures.choose(rows, setting, entries[entry]) for name, entry, _, _ in METHODS}
    scores = {name: read_scores(run_dir(args.dataset, args.split, method, args.results)) for name, method in methods.items()}
    alarms = {name: calibrate(kind, label, methods[name], scores[name], split, inputs.episodes, inputs.window) for name, _, kind, label in METHODS}
    rates = realised_rates(alarms, scores, split, inputs)
    print(*check_report(alarms, rates), sep="\n")

    layers = {layer.name: load_points(embeddings.points_path(args.results, args.dataset, args.split, methods["ours"], layer)) for layer in embeddings.LAYERS}
    with np.load(run_dir(args.dataset, args.split, methods["ours"], args.results) / "mode_logits.npz") as z:
        mode_logits = {k: z[k] for k in keys}
    step_seconds = spec.replan_steps / spec.control_hz

    signatures = signature_table(spec.data_dir / "splits" / "splits.csv")
    records = []
    for number, key in enumerate(keys, start=1):
        eid, episode, row = f"ep{number}", inputs.episodes[key], instruction_of(args.dataset, key)
        video = videos.prepare(args.videos, key, args.out / "media" / "rollouts" / eid, f"media/rollouts/{eid}", int(row["n_control"]) / spec.control_hz)
        mode = inputs.targets.modes.get(key)
        record = episodes.episode_record(eid, key, episode, row["prompt"], mode, step_seconds, inputs.window, alarms, scores, mode_logits[key],
                                         {name: embeddings.episode_path(points, key) for name, points in layers.items()}, video)
        signature, held_out = signatures.get((row["level"], row["task"]), (None, False))
        record["callouts"] = callouts_for(args.callouts, key, video["duration"])
        record["signature"] = signature
        record["signature_unseen"] = bool(signature) and held_out and group_of(args.split) == "signature"  # only in a leave-signature-out split is a held-out signature unseen
        for name in ("scores", "threshold"):  # every per-step list has one entry per policy step
            for method in record["methods"].values():
                assert not isinstance(method[name], list) or len(method[name]) == episode.n_steps, f"{key}: {name}"
        assert len(record["mode_probs"]) == episode.n_steps and all(len(v) == episode.n_steps for v in record["embeddings"].values()), key
        write_json(args.out / "data" / "episodes" / f"{eid}.json", record)
        records.append(record)

    for layer in embeddings.LAYERS:
        write_json(args.out / "data" / "embeddings" / f"{layer.name}.json", embeddings.summary(layers[layer.name], split, layer, args.seed))
    write_json(args.out / "data" / "index.json", {
        "version": content_version(args.out),
        "dataset": args.dataset,
        "split": args.split,
        "unseen_episodes": len(split.unseen),
        "step_seconds": step_seconds,
        "mode_names": list(MODE_NAMES),
        "operating_point": {"fpr": FPR_TARGET, "tpr_early": TPR_EARLY_TARGET, "calibration_episodes": len(split.calibration)},
        "unseen_kind": UNSEEN_KIND.get(group_of(args.split), "held-out tasks"),
        "selection": ({"keys": keys, "criteria": args.criteria, "from": "the unseen episodes of the split"} if args.keys else
                      {"seed": args.seed, "successes": args.successes, "failures": args.failures, "from": "the unseen episodes of the split"}),
        "methods": {name: {"label": a.label, "configuration": a.method, "calibration": a.calibration, "certified": a.threshold is not None, "rates": rates.get(name)}
                    for name, a in alarms.items()},
        "layers": [{"name": layer.name, "title": layer.title, "file": f"data/embeddings/{layer.name}.json"} for layer in embeddings.LAYERS],
        "episodes": [{"id": r["id"], "file": f"data/episodes/{r['id']}.json", "instruction": r["instruction"], "success": r["success"], "mode": r["mode"]} for r in records],
    }, indent=2)
    print(f"wrote {len(records)} episodes, {len(embeddings.LAYERS)} layers and their videos under {args.out}/")


if __name__ == "__main__":
    main()
