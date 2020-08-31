import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark capsule parallelism')

    # noinspection PyTypeChecker
    parser.add_argument('-c', '--capsule-dir', type=Path, required=True,
                        help="Directory of unpackaged capsules to be tested")
    parser.add_argument('-w', '--num-workers', type=int, nargs="+",
                        required=True,
                        help="Number of threads to parallelize to. In bash "
                             "you can use {0..10} or `seq 0 2 10` to avoid "
                             "having to list a lot of values.")
    parser.add_argument('-s', '--num-samples', type=int, required=True,
                        help="Number of samples to run for each capsule")

    # noinspection PyTypeChecker
    parser.add_argument('-f', '--output-csv', type=Path, default=None,
                        help="If specified, results will be written to the "
                             "provided path in .csv format")
    # noinspection PyTypeChecker
    parser.add_argument('-g', '--output-graph', type=Path, default=None,
                        help="If specified, results will be drawn to a graph "
                             "and saved at the provided path as an image."
                             "Example: output.html")

    return parser.parse_args()
