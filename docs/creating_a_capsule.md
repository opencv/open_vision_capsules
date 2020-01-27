For application developers, OpenVisionCapsules provides a function to package up
unpackaged capsules. The optional `key` field encrypts the capsule with AES.

```python
from vcap import package_capsule

package_capsule(Path("detector_person"),
                Path("capsules", "detector_person.cap"),
                key="[AES Key]")
```

For capsule developers for an application, it is the job of the application to
provide a way to package capsules. Please see the documentation for the
application you are using for more information.

## Creating an Object Detector Capsule with Supervisely

If you’ve trained tensorflow-object-detection-API object detector using
Supervisely, you can follow the following steps to deploy your model as a
capsule:

### Set up the TF Object Detection API

First, set up the Tensorflow Object Detection API on your machine by cloning
the `tensorflow/models` repository and following the object detection API
installation instructions. Make sure the tests pass before continuing-
otherwise, you might have forgotten to set up certain environment variables!

### Freeze your trained Supervisely model

Next, you must download your trained Supervisely model and extract it. Inside
you should see the following directory structure:

```
    <DOWNLOADED MODEL DIR>
    ├── config.json
    ├── model.config
    └── model_weights
        ├── checkpoint
        ├── model.ckpt.data-00000-of-00001
        ├── model.ckpt.index
        └── model.ckpt.meta
```

Now, you must simply freeze the model to get the `frozen_inference_graph.pb`.
To do that, run `models/research/object_detection/export_inference_graph.py`
script inside of your downloaded model directory.

```bash
python PATH/TO/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path model.config \
    --trained_checkpoint_prefix model_weights/model.ckpt \
    --output_directory .
```

There should now be a frozen_inference_graph.pb in the current directory. This
is the model file that has been optimized for inference, and is much more
portable for production use. 
