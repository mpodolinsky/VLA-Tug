"""python -m cli.train <dataset> --methods all|name,name --splits all|name,name --seed 0

Train every selected method on every selected split and save its weights under results/.
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--methods", default="all")
    p.add_argument("--splits", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
