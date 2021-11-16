import argparse
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import List, NoReturn, Optional, Tuple
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd

from capsule_infer.capsule_infer import capsule_inference, parse_images, capsule_infer_add_args, \
    parse_capsule_info


class StoreDictKeyPair(argparse.Action):
     def __init__(self, option_strings, dest, nargs=None, **kwargs):
         self._nargs = nargs
         super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         print("values: {}".format(values))
         for kv in values:
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)


def output_report(output_filename, cmdline, detection_results, data_detection, data_attribute, data_truth, true_threshold, false_threshold):
    # May need to do wild match so it does not depend on the data_attribute
    confidence_key = f'{data_attribute}_confidence'
    if detection_results:
        if detection_results[0].attributes and detection_results[0].extra_data:
            results = sorted(detection_results, key=lambda d: d.extra_data[confidence_key])
            confidence_true = []
            confidence_false = []
            confidence_unknown = []
            for result in results:
                confidence = result.extra_data[confidence_key]
                if 'true' in result.attributes[data_attribute]:
                    true_attribute_label = result.attributes[data_attribute]
                    confidence_true.append(confidence)
                elif 'false' in result.attributes[data_attribute]:
                    false_attribute_label = result.attributes[data_attribute]
                    confidence_false.append(confidence)
                else: # 'unknown'
                    unknown_attribute_label = result.attributes[data_attribute]
                    confidence_unknown.append(confidence)

            true_in_all = len(confidence_true) / (len(results))
            false_in_all = len(confidence_false) / (len(results))
            unknown_in_all = len(confidence_unknown)/len(results)

            fig, (plt_true, plt_false, plt_unknown) = plt.subplots(3)
            fig.suptitle(f'detection: {data_detection}, attribute: {data_attribute}, data_truth: {data_truth}')

            true_report, true_max_bin = create_report(plt_true, confidence_true, true_attribute_label, true_in_all, data_truth=='true')
            false_report, false_max_bin = create_report(plt_false, confidence_false, false_attribute_label, false_in_all, data_truth=='false')
            unknown_report, unknown_max_bin = create_report(plt_unknown, confidence_unknown, unknown_attribute_label, unknown_in_all, False)

            threshold_text = f'true_threshold: {true_threshold:.2%}, false_threshold: {false_threshold:.2%}'
            if data_truth=='true':
                correct_in_none_unknown = len(confidence_true) / (len(confidence_true) + len(confidence_false))
                plt_true.text(0, true_max_bin * 0.8, f'Counts in none unknown: {correct_in_none_unknown:.2%}', color='green')
                plt_true.text(0, true_max_bin * 0.6, threshold_text, color='green')
            else:
                correct_in_none_unknown = len(confidence_false) / (len(confidence_true) + len(confidence_false))
                plt_false.text(0, false_max_bin * 0.8, f'Counts in none unknown: {correct_in_none_unknown:.2%}', color='green')
                plt_false.text(0, false_max_bin * 0.6, threshold_text, color='green')

            output_filename = output_filename + "_" + data_truth
            with open(output_filename + '.txt', 'w') as f:
                f.write(cmdline + "\n\n")
                f.write(true_report + "\n\n")
                f.write(false_report + "\n\n")
                f.write(unknown_report)
                f.close()
            fig.savefig(output_filename + '.png')


def create_report(plt, confidence, prediction_label, percentage, correct_as_green):
    if correct_as_green:
        color = 'green'
    else:
        color = 'blue'
    title = f'{len(confidence)} {prediction_label}: {percentage:.2%}'

    bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    y, x, _ = plt.hist(confidence, bins=bins, color=color)
    plt.set(xlabel='confidence')
    plt.set(ylabel=f'counts')
    plt.set_title(title)
    plt.label_outer()

    data = pd.Series(confidence, name=title)
    report_table = data.value_counts(bins=bins, sort=False)

    report_text = report_table.to_markdown()

    return report_text, y.max()


def add_args(parser) -> Tuple[Path, Optional[Path], List[Path]]:

    capsule_infer_add_args(parser)

    parser.add_argument(
        "--images-true",
        type=Path,
        nargs="+",
        help="The test images classified as true. "
        "Paths to one or more images to run inference on. If the path is a "
        "directory, then *.png or *.jpg images in the directory will be used.",
    )
    parser.add_argument(
        "--images-false",
        type=Path,
        nargs="+",
        help="The test images classified as false. "
        "Paths to one or more images to run inference on. If the path is a "
        "directory, then *.png or *.jpg images in the directory will be used.",
    )
    parser.add_argument(
        "--data",
        dest="data_dict",
        action=StoreDictKeyPair,
        nargs="+",
        required=False,
        help="Indicating the ground truth of the data input, defaults/examples are "
        "detection=person attribute=helmet true_threshold=0.25 false_threshold=0.25"
    )

    return


def read_cmdline():
    cmdline = '';

    for arg in sys.argv[1:]:
        if ' ' in arg:
            cmdline += '"{}"  '.format(arg)
        else:
            cmdline += "{}  ".format(arg)

    return cmdline


def main():
    parser = ArgumentParser(
        description="A helpful tool for running inference and generate accuracy benchmarking report on a capsule."
    )

    add_args(parser)

    args = parser.parse_args()

    cmdline = read_cmdline()

    data_detection, data_attribute = 'person', 'helmet',
    true_threshold, false_threshold = 0.0, 0.0
    if args.data_dict:
        if 'detection' in args.data_dict:
            data_detection = args.data_dict['detection']

        if 'attribute' in args.data_dict:
            data_attribute = args.data_dict['attribute']

        if 'true_threshold' in args.data_dict:
            true_threshold = float(args.data_dict['true_threshold'])

        if 'false_threshold' in args.data_dict:
            false_threshold = float(args.data_dict['false_threshold'])

    packaged_capsule_path, unpackaged_capsule_path, capsule_name = parse_capsule_info(args)

    output_filename_prefix = capsule_name + "_" + data_detection + "_" + data_attribute\
                           + "_T" + str(true_threshold) + "_F" + str(false_threshold)
    if args.images:
        images = parse_images(args.images)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection,
                                              true_threshold, false_threshold)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, data_attribute, "", true_threshold, false_threshold)

    if args.images_true:
        images = parse_images(args.images_true)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection,
                                              true_threshold, false_threshold)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, data_attribute, "true", true_threshold, false_threshold)

    plt.waitforbuttonpress(1)

    if args.images_false:
        images = parse_images(args.images_false)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection,
                                              true_threshold, false_threshold)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, data_attribute, "false", true_threshold, false_threshold)

    plt.waitforbuttonpress(1)
    print("Press Enter to EXIT")
    input()

    return


if __name__ == "__main__":
    main()
