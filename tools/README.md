# How to Build

```shell
python3 compile.py build_ext build_binaries 
```
Clean up a build,

```shell
python3 compile.py clean 
```

# Capsule Infer

This tool is used during development to quickly and easily run inference 
with a capsule. 

## Usage
Make sure you are running in an environment with `vcap` and `vcap-utils` 
installed.

Running is simple. Point the script to the capsule of choice, and list one 
or more images to run inference on. For example:
```shell
python3 capsule_infer.py --capsule my-detector-capsule/ --images img.png 
img2.png 
```

You can also point `--images` to a path that contains images.

# Capsule Classifier accuracy

This tool is used during development to quickly and easily run accuracy benchmark 
with a capsule. The output is a set of benchmark reports.

## Usage
Make sure you are running in an environment with `vcap` and `vcap-utils` 
installed.

Running is simple. Point the script to the capsule of choice, and list one 
or more images to run inference on. For example:
```shell
python3 capsule_classifier_accuracy.py --capsule <capsule directory or .cap> --images-true <true image directory> --images-false <false image directory> --data attribute=<name> true_threshold=0.0 false_threshold=0.0 detection=<class name>
```

Optional image categories are as follows,

 `--images`, `--images-true`, `--images-false`
