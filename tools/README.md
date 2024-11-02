# How to Build

```shell
python3 compile.py build_ext build_binaries 
```
Clean up a build,

```shell
python3 compile.py clean 
```

# Capsule Inference

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

# Build the module
Build the environment first,
```shell
DOCKER_BUILDKIT=1 docker build --file tools/Dockerfile.build-env --build-arg UBUNTU_VERSION=20.04 --tag open_vision_capsule_env --no-cache .
```
Then build the module
```shell
docker run -it --rm -v ./tools:/open_vision_capsules/tools -w /open_vision_capsules/tools open_vision_capsule_env:latest bash -c "poetry build"
```
Build an Ubuntu image for testing the module
```shell
DOCKER_BUILDKIT=1 docker build --file tools/Dockerfile.ubuntu --build-arg UBUNTU_VERSION=20.04 --tag ubuntu-2004 .
```
Run the Ubuntu image for testing the module
```shell
docker run -it --rm -v ./tools:/open_vision_capsules/tools -w /open_vision_capsules/ ubuntu-2004 bash
```
Then
```shell
pip install tools/dist/openvisioncapsule_tools-0.3.8-py3-none-any.whl
openvisioncapsule-tools
```


