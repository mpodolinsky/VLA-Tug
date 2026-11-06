# site_export

Writes the data of the project page's interactive figure (`website/`) from the saved runs. Nothing is trained: it reads scores, calibrates thresholds
and exports plain JSON and the rollout videos.

```
site_export/
  export.py       the command line: draws the episodes, calls the modules below, writes website/data and website/media
  calibrate.py    the alarm thresholds of the three methods at one operating point
  episodes.py     the random draw or the hand-picked list of episodes, and the page's record of one episode
  embeddings.py   the principal-axis views (three layers of the probe): sampled points, class Gaussians, an episode's path
  videos.py       checks the videos (browser-ready, length) and copies them next to the page
  main_table.py   the paper's main table (means with standard deviations, and the number of runs per setting) as HTML, written into the Results section
```

## What the figure shows

For one split of one dataset (default: `task_seed0` of LIBERO-X, whose unseen episodes are new tasks; the page uses `signature_seed2`, whose unseen
episodes are new task signatures), for **unseen** episodes that are either drawn at random (default: seed 0, one success and two failures;
`episodes.select`) or listed by hand with `--keys` (`episodes.chosen`, which refuses an episode that is not unseen):

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
python -m site_export.export --list          # the episodes that are drawn (or the --keys), and the command that fetches their videos
python -m site_export.export --videos <folder with <episode key>/rollout.mp4 and wrist.mp4>
```

`--dataset`, `--split`, `--seed`, `--successes`, `--failures`, `--keys`, `--criteria`, `--results` and `--out` change the draw and the folders. The videos of the
released datasets are listed by `--list`; their pooled features and annotations are what the saved runs were trained on. The principal axes of a split
come from `python -m cli.tsne --embedding probe --reducer pca --fit-set train_pre --split <split> --layer hidden|score_hidden|mode_hidden --out results/pca3_train`
(one run per layer), which has to be run once for every split the page shows.

### Callouts on the score timeline

`callouts.json` ({episode key: [{"time": seconds, "text": ...}]}) holds the numbered notes of an episode. The export copies them, sorted by time, into the episode's
record (`--callouts` names another file; a time outside the video or an empty text is an error); the page draws a numbered disc at each time in a strip above the
score rows, with a faint line down to the time axis, and lists the notes with their text above the timeline. Notes the video has already reached are drawn in full
colour. An episode without notes shows neither. Times of alarms in a note should equal the alarm times the page prints (they come from the saved scores).

### The episodes of the page

The page shows three episodes of `signature_seed2` that were **selected**, not drawn: a success in which VLA-Tug stays silent while SAFE-LSTM raises a false
alarm (STAC stays silent too), and two failures that VLA-Tug flags as they start or before: in the first STAC and SAFE-LSTM flag the failure much later, in the
second STAC never flags it and SAFE-LSTM flags it first, at 2.4 s. They were found by listing, for every unseen episode of the LIBERO-X splits, the outcome of each
method at its calibrated threshold, and then looking at the videos: episodes in which an object flies away, is knocked off or is carried by the gripper by mistake
(simulator glitches or erratic motion) were left out, and so were successes that end before the task is visibly done. They illustrate the signal and are not typical:
over all 560 unseen episodes of the split the three methods give the false-positive and early true-positive rates the page prints under the figure. The page says
this under the figure (`--criteria`). The command is

```bash
python -m site_export.export --split signature_seed2 --videos <folder> \
  --keys "LEVEL1/44_EXTENSION_KITCHEN_SCENE42_place_the_red_mug_in_the_basket/ep008" \
         "LEVEL3/32_EXTENSION_KITCHEN_SCENE32_LEVEL3__T186_turn_off_the_flat_stove/ep007" \
         "LEVEL3/63_EXTENSION_KITCHEN_SCENE9_LEVEL3__T55_place_the_black_bowl_to_the_right_of_the_bowl_drainer/ep008" \
  --criteria "one success in which VLA-Tug stays silent while SAFE-LSTM raises a false alarm (at 4.4 s; STAC stays silent too), and two failures that VLA-Tug flags as they start or before (at 4.4 s and 14.4 s): in the first STAC and SAFE-LSTM flag the failure much later (at 32.8 s and 22.8 s), in the second STAC never flags it and SAFE-LSTM flags it first, at 2.4 s."
```

## Output

```
website/data/index.json                one record: the operating point, the methods, the layers, the episodes, a content version
                                       (the page adds it to the URL of every file, so that a new export with the same file names is not served from the cache)
website/data/episodes/<id>.json        one episode: per-step scores, thresholds, alarm steps, mode probabilities, embedding paths, video info, the task's signature
                                       (from data/<dataset>/splits/splits.csv, LIBERO-X only) and whether it is an unseen signature
website/data/embeddings/<layer>.json   one layer: class Gaussians and a sample of training points
website/media/rollouts/<id>/           agent.mp4 and wrist.mp4
```

Every per-step list has one entry per policy step, and the video length is checked against the episode's control steps, because the page maps video
time to policy steps with `step_seconds` (replan steps / control rate).

## The main table

```bash
python -m site_export.main_table          # reads results/figures/table_main_std.tex (python -m cli.figures), rewrites the table in website/index.html
```

The page takes the paper's table itself, so every mean, every standard deviation and every bold (best) and italic (second best) mark is the paper's; only the markup changes (our row is shaded and called VLA-Tug). The row under the setting names says how many runs a setting averages (seeds, or folds for the skill split, or seeds x folds on the real robot); it is counted from the splits found under `results/`, not typed in. The table lies between two comment markers of `website/index.html`; running the command again replaces what is between them.

Tests: `pytest tests/test_site_export.py`.
