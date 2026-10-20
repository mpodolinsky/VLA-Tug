"""python -m cli.prepare <dataset> <rollout_root>

Pool raw rollouts into data/<dataset>/features.npz. The only command that reads raw rollouts.
"""

import argparse

from fp.data.pooling import prepare_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("rollout_root")
    a = p.parse_args()
    prepare_dataset(a.dataset, a.rollout_root)


if __name__ == "__main__":
    main()
