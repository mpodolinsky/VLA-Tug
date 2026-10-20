"""python -m cli.evaluate <dataset> --methods all|name,name

Score, calibrate (SAFE band and max-conformal; LTT separately) and write curves and summaries under results/.
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--methods", default="all")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
