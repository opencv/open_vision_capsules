# Capsule Infer

This tool is used during development to quickly and easily run accuracy benchmark 
with a capsule. The output is a set of benchmark reports.

## Usage
Make sure you are running in an environment with `vcap` and `vcap-utils` 
installed.

Running is simple. Point the script to the capsule of choice, and list one 
or more images to run inference on. For example:
```shell
python3 capsule_infer.py --capsule my-classifier-capsule/ --images-true my_dataset_true_positive/ --images-false my_dataset_true_negative/ --data detection=my_detection attribute=my_attribute true_threshold=0.0 false_threshold=0.0
```

Optional image categories are as follows,

 `--images`, `--images-true`, `--images-false`
