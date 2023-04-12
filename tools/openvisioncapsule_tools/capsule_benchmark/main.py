import pandas as pd

import output
from args import parse_args
from benchmarking import BenchmarkSuite


def main():
    args = parse_args()

    testing_suite = BenchmarkSuite(args.capsule_dir, args.parallelism)
    results = testing_suite.test(args.num_samples)

    df = pd.DataFrame.from_records(results,
                                   columns=BenchmarkSuite.Result._fields)
    df.sort_values(by=list(df.columns), inplace=True, ignore_index=True)

    output.generate_output(
        output=df,
        csv_path=args.output_csv,
        graph_path=args.output_graph
    )


if __name__ == '__main__':
    main()
