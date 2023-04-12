import datetime
import json
import os
from pathlib import Path

from tools.bulk_accuracy.capsule_accuracy_common import DetectionResult


def write_to_file(output_directory, lines):
    output_file_path = Path(output_directory, f'analysis_detection_result_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_file_path), "w") as fp:
        fp.writelines(lines)


def output_detection_result_csv(output_directory, data_list):
    title = (
        f"sample_path,offset_time,img_id,classified_type,confidence_value{os.linesep}"
    )
    lines = [title]

    for e in data_list:
        dd: DetectionResult = e
        line = f"{dd.sample_path},{dd.offset_time},{dd.img_id},{dd.classified_type},{dd.confidence_value}{os.linesep}"
        lines.append(line)

    write_to_file(output_directory, lines)


def output_detection_result_json(output_directory, data_list):
    lines = []
    for e in data_list:
        dd: DetectionResult = e
        line = dd.to_json_str()
        lines.append(f"{line}{os.linesep}")

    write_to_file(output_directory, lines)


def load_detection_result_json(input_json_file):
    data_list = []
    with open(input_json_file, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        json_obj = json.loads(line)
        data_list.append(DetectionResult.from_json(json_obj))
    return data_list


def load_detection_result_csv(input_csv_file):
    data_list = []
    with open(input_csv_file, "r") as fp:
        lines = fp.readlines()
    for idx in range(1, len(lines)):
        line = lines[idx]
        data_list.append(DetectionResult.from_csv(line))
    return data_list


def sum_classified_data(detected_data_list):
    class GroupedDetectionResult:
        def __init__(self, classified_type, threshold_confidence_value):
            self.classified_type = classified_type
            self.threshold_confidence_value = threshold_confidence_value
            self.num_target_yes = 0
            self.num_target_no = 0
            self.num_unknown = 0

        def accumulate(self, rst: DetectionResult):
            if rst.classified_type == self.classified_type:
                if rst.confidence_value >= self.threshold_confidence_value:
                    self.num_target_yes += 1
                else:
                    self.num_unknown += 1
            else:
                self.num_target_no += 1

        def to_string(self):
            return f"{self.classified_type}\t{format(self.threshold_confidence_value, '.2f')}\t{self.num_target_yes}\t{self.num_unknown}\t{self.num_target_no}"

    all_true, all_false = [], []
    for threshold in range(20):
        sub_sum_true, sub_sum_false = GroupedDetectionResult(True, 0.05 * threshold), GroupedDetectionResult(False, 0.05 * threshold)
        for detection in detected_data_list:
            sub_sum_true.accumulate(detection)
            sub_sum_false.accumulate(detection)
        all_true.append(sub_sum_true)
        all_false.append(sub_sum_false)

    print("True=>")
    for e in all_true:
        print(e.to_string())

    print("False=>")
    for e in all_false:
        print(e.to_string())


if __name__ == "__main__":
    data = load_detection_result_csv("/home/leefr/temp/analysis-data/analysis_data_20220623005435.csv")
    sum_classified_data(data)
