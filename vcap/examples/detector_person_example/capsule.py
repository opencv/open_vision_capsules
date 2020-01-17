from vcap import (
    BaseCapsule,
    DeviceMapper,
    NodeDescription,
    FloatOption,
    BoolOption,
    IntOption)
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_example"
    version = 1
    device_mapper = DeviceMapper.map_to_all_gpus()
    # This capsule takes no input from other capsules
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    # This capsule produces DetectionNodes of people
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = lambda capsule_files, device: Backend(
        model_bytes=capsule_files["ssd_mobilenet_v1_coco.pb"],
        metadata_bytes=capsule_files["dataset_metadata.json"],
        device=device)
    options = {
        "detection_threshold": FloatOption(
            description="The confidence threshold for the model. A higher "
                        "value means fewer detections",
            default=0.5,
            min_val=0.1,
            max_val=1.0),
        "scale_frame": BoolOption(
            description="If true, the frame width and height will be clamped "
                        "to the value of scale_frame_max_side_length, "
                        "preserving aspect ratio",
            default=False),

        "scale_frame_max_side_length": IntOption(
            description="The width or height to scale frames down to "
                        "if scale_frames is True",
            default=2000,
            min_val=200,
            max_val=4000)
    }
