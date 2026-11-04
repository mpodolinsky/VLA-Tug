# site_export

Writes the data of the project page's interactive figure (`website/`) from the saved runs. Nothing is trained: it reads scores, calibrates thresholds
and exports plain JSON and the rollout videos.

```
site_export/
  export.py       the command line: draws the episodes, calls the modules below, writes website/data and website/media
  calibrate.py    the alarm thresholds of the three methods at one operating point
  episodes.py     the random draw of the episodes, and the page's record of one episode
  embeddings.py   the principal-axis views (three layers of the probe): sampled points, class Gaussians, an episode's path
  videos.py       checks the videos (browser-ready, length) and copies them next to the page
```

## What the figure shows

For one split of one dataset (default: `task_seed0` of LIBERO-X, whose unseen episodes are new tasks), for episodes drawn at random from its
**unseen** episodes (default: seed 0, one success and two failures; the draw is `episodes.select`):

* the probe's state in its first three principal axes, for three layers (LSTM state, failure-score head, failure-mode head), with a sample of the
  training steps, one Gaussian per class and the path of the chosen episode;
* the scores of three methods over time with their calibrated thresholds, the alarm and the annotated failure time;
* the predicted failure mode (top three) at every step.

The three methods are the configurations of the paper's main table for the split's setting (`fp.eval.figures.choose`), and each is calibrated on
the split's calibration episodes only, at one operating point (`calibrate.py`):

| method | calibrator | target |
|---|---|---|
| VLA-Tug | Learn-then-Test (constant threshold) | false-positive rate <= 15% and early true-positive rate >= 15%, confidence 90% |
| SAFE-LSTM | SAFE's functional conformal band (varies with the step) | false-positive rate level 15% |
| STAC | max-conformal (constant threshold) | false-positive rate level 15% |

`export.py` prints each threshold and the false-positive and early true-positive rates it gives on the calibration episodes and on all unseen
episodes of the split, and stores them in `index.json`; the page shows them under the figure.

## Run

Run everything from the root of a checkout that has `data/` and `results/` (the datasets and the saved runs, see the main README), after
`pip install -e ".[dev]"`; `ffmpeg` and `ffprobe` are needed for the videos.

```bash
python -m site_export.export --list          # the episodes that are drawn, and the command that fetches their videos
python -m site_export.export --videos <folder with <episode key>/rollout.mp4 and wrist.mp4>
```

`--dataset`, `--split`, `--seed`, `--successes`, `--failures`, `--results` and `--out` change the draw and the folders. The videos of the released
datasets are listed by `--list`; their pooled features and annotations are what the saved runs were trained on.

## Output

```
website/data/index.json                one record: the operating point, the methods, the layers, the episodes
website/data/episodes/<id>.json        one episode: per-step scores, thresholds, alarm steps, mode probabilities, embedding paths, video info
website/data/embeddings/<layer>.json   one layer: class Gaussians and a sample of training points
website/media/rollouts/<id>/           agent.mp4 and wrist.mp4
```

Every per-step list has one entry per policy step, and the video length is checked against the episode's control steps, because the page maps video
time to policy steps with `step_seconds` (replan steps / control rate).

Tests: `pytest tests/test_site_export.py`.
