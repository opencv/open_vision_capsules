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

## Limitations
This tool only works with capsules that don't require an input (such as 
detectors). Capsules that require a DetectioNode as input (such as 
classifiers, encoders) don't currently work with this tool. 

