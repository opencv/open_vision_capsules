import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

# from tools.capsule_infer.capsule_infer import capsule_inference, parse_images, capsule_infer_add_args, \
#     parse_capsule_info
#
# from tools.capsule_infer.capsule_infer import read_options, capsule_options_and_key
from tools.capsule_infer.capsule_infer import capsule_inference, parse_images, capsule_infer_add_args, \
    parse_capsule_info, read_options, capsule_options_and_key


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        print("values: {}".format(values))
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def output_report(output_filename, cmdline, detection_results, data_detection, attribute_name, data_truth, true_threshold, false_threshold):
    if detection_results:
        if detection_results[0].attributes and detection_results[0].extra_data:
            confidence_key = attribute_name + '_confidence'
            # detections = sorted(detection_results, key=lambda d: d.extra_data[confidence_key])
            detections = detection_results
            confidence_true = []
            confidence_false = []
            confidence_unknown = []
            true_attribute_label = ""
            false_attribute_label = ""
            unknown_attribute_label = ""
            for det in detections:
                if confidence_key in det.extra_data:
                    confidence = det.extra_data[confidence_key]
                else:
                    confidence = det.extra_data['confidence']
                if attribute_name in det.attributes:
                    if 'true' in det.attributes[attribute_name]:
                        true_attribute_label = det.attributes[attribute_name]
                        confidence_true.append(confidence)
                    elif attribute_name in det.attributes and 'false' in det.attributes[attribute_name]:
                        false_attribute_label = det.attributes[attribute_name]
                        confidence_false.append(confidence)
                    else:  # 'unknown'
                        unknown_attribute_label = det.attributes[attribute_name]
                        confidence_unknown.append(confidence)
                elif 'unknown' in det.attributes:
                    unknown_attribute_label = det.attributes['unknown']
                    confidence_unknown.append(confidence)

            true_in_all = len(confidence_true) / (len(detections))
            false_in_all = len(confidence_false) / (len(detections))
            unknown_in_all = len(confidence_unknown) / len(detections)

            fig, (plt_true, plt_false, plt_unknown) = plt.subplots(3)
            params = {'axes.labelsize': 'medium',
                      'axes.titlesize': 'medium',
                      'figure.titlesize': 'medium'}
            plt.rcParams.update(params)
            threshold_text = f'true_threshold: {true_threshold:.2%}, false_threshold: {false_threshold:.2%}'
            fig.suptitle(f'detection: {data_detection}, attribute: {attribute_name}, data_truth: {data_truth}\n' + threshold_text)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

            correct_in_none_unknown = 0
            len_none_unknown = len(confidence_true) + len(confidence_false)
            if len_none_unknown != 0:
                if data_truth == 'true':
                    correct_in_none_unknown = len(confidence_true) / len_none_unknown
                elif data_truth == 'false':
                    correct_in_none_unknown = len(confidence_false) / len_none_unknown

            true_report_title = f'True: {true_attribute_label}: {true_in_all:.2%}, in none unknown: {correct_in_none_unknown:.2%}'
            false_report_title = f'False: {false_attribute_label}: {false_in_all:.2%}, in none unknown: {correct_in_none_unknown:.2%}'
            unknown_report_title = f'Unknown: {unknown_attribute_label}: {unknown_in_all:.2%}'

            true_report, true_max_bin = create_report(plt_true, confidence_true, true_report_title, data_truth == 'true')
            false_report, false_max_bin = create_report(plt_false, confidence_false, false_report_title, data_truth == 'false')
            unknown_report, unknown_max_bin = create_report(plt_unknown, confidence_unknown, unknown_report_title, False)

            output_filename = output_filename + "_" + data_truth
            with open(output_filename + '.txt', 'w') as f:
                f.write(cmdline + "\n\n")
                f.write(true_report + "\n\n")
                f.write(false_report + "\n\n")
                f.write(unknown_report)
                f.close()
            fig.savefig(output_filename + '.png')


def create_report(plt, confidence, title_text, correct_as_green):
    if correct_as_green:
        color = 'green'
    else:
        color = 'blue'
    title = f'{len(confidence)} {title_text}'
    if len(confidence) == 0:
        plt.set_ylim([0, 1])

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


def add_args(parser):
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
    parser.add_argument(
        "--nowait",
        dest="nowait",
        action="store_true",
        default=False,
        help="Wait for keypress before exit. Default: %(default)s",
    )


def read_cmdline():
    cmdline = ''

    for arg in sys.argv[1:]:
        if ' ' in arg:
            cmdline += '"{}"  '.format(arg)
        else:
            cmdline += "{}  ".format(arg)

    return cmdline


def classifier_accuracy():
    parser = ArgumentParser(
        description="A helpful tool for running inference and generate accuracy benchmarking report on a capsule."
    )

    add_args(parser)

    args = parser.parse_args()

    cmdline = read_cmdline()

    data_detection, attribute_name = 'person', 'helmet'
    true_threshold, false_threshold = 0.0, 0.0
    if args.data_dict:
        print(f'++++++args.data_dict={args.data_dict}+++++++++')

        if 'detection' in args.data_dict:
            data_detection = args.data_dict['detection']

        if 'attribute' in args.data_dict:
            attribute_name = args.data_dict['attribute']

        if 'true_threshold' in args.data_dict:
            true_threshold = float(args.data_dict['true_threshold'])

        if 'false_threshold' in args.data_dict:
            false_threshold = float(args.data_dict['false_threshold'])

    packaged_capsule_path, unpackaged_capsule_path, capsule_name = parse_capsule_info(args)

    output_filename_prefix = f'{capsule_name}_{data_detection}_{attribute_name}' \
                             f'_T{true_threshold}_F{false_threshold}'

    input_options, capsule_key = capsule_options_and_key(args)

    if args.nowait is not True:
        wait_time = 1
    else:
        wait_time = None

    if args.images:
        images = parse_images(args.images)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection, input_options, capsule_key, wait_time=wait_time)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, attribute_name, "", true_threshold, false_threshold)

    if args.images_true:
        images = parse_images(args.images_true)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection, input_options, capsule_key, wait_time=wait_time)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, attribute_name, "true", true_threshold, false_threshold)

    if args.nowait is not True:
        plt.waitforbuttonpress(1)

    if args.images_false:
        images = parse_images(args.images_false)
        detection_results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path,
                                              images, data_detection, input_options, capsule_key, wait_time=wait_time)
        output_report(output_filename_prefix, cmdline, detection_results, data_detection, attribute_name, "false", true_threshold, false_threshold)

    if args.nowait is not True:
        plt.waitforbuttonpress(0)

    return


if __name__ == "__main__":
    classifier_accuracy()
