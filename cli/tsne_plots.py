"""python -m cli.tsne_plots --points results/tsne/liberox_groot/vla/points.npz [--split task_seed0] [--colorings task skill outcome mode outcome_lines mode_lines] [--out DIR]

Draws the figures of saved t-SNE points (python -m cli.tsne), one PNG per colouring and subset (all, seen, seen_held_out, unseen), all on the same axes, without repeating the
t-SNE: <points dir>/figures/<split>/<coloring>__<subset>.png. --split is the split whose seen and unseen episodes the subsets are (the probe's own for a probe; any for
the VLA embedding). Colourings: task, skill, outcome (success: light to dark blue; failure: light blue to red at the failure step), mode (failure: light blue to the colour of
its failure mode); outcome_ellipses_late/_all and mode_ellipses_late/_all fit a Gaussian to each class (fp/eval/class_fits.py) and draw its one-standard-deviation ellipse;
outcome_gaussians_late/_all and mode_gaussians_late/_all draw the same ellipses without the points; outcome_ellipses_path and mode_ellipses_path add the path of the
most typical successful episode as a black arrow, outcome_ellipses_paths and mode_ellipses_paths also that of a failure of --path-mode (dashed, ending in a cross);
outcome_ellipses_pairs (only when asked for) draws --pairs figures per subset, each with a random success and a random failure of any mode; outcome_3d_ellipsoids, outcome_3d_gaussians,
mode_3d_ellipsoids and mode_3d_gaussians draw 3D Gaussians as ellipsoids with and without the points;
outcome_3d and mode_3d draw the first three principal axes of PCA points from three directions;
outcome_lines and mode_lines draw a sample of episodes as lines in those colours over the other points in grey, to follow the progression of an
episode. See fp/eval/tsne_plots.py.
"""

import argparse
from pathlib import Path

from fp.data.targets import MODE_NAMES
from fp.eval import tsne_plots


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--split")
    p.add_argument("--colorings", nargs="+", choices=tsne_plots.FIGURES + tuple(tsne_plots.PAIRS))
    p.add_argument("--seed", type=int, default=0, help="the seed of the random episodes of outcome_ellipses_pairs and of the order of the points; another seed draws other pairs (use another --out to keep both)")
    p.add_argument("--pairs", type=int, default=tsne_plots.N_PAIRS, help="the number of random pairs of a success and a failure drawn by outcome_ellipses_pairs")
    p.add_argument("--out", type=Path)
    p.add_argument("--path-mode", choices=MODE_NAMES, default=tsne_plots.PATH_MODE, help="the failure mode of the failure drawn by outcome_ellipses_paths and mode_ellipses_paths")
    args = p.parse_args()
    if tsne_plots.load_points(args.points)["config"]["embedding"] == "vla" and not args.split:
        p.error("the VLA embedding has no split of its own: pass --split")
    for path in tsne_plots.render(args.points, args.split, args.out, args.colorings, seed=args.seed, path_mode=args.path_mode, n_pairs=args.pairs):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
