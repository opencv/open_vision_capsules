from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper
)
from .backend import Backend
from . import config


class Capsule(BaseCapsule):
    name = "classifier_gait_example"
    version = 1
    device_mapper = DeviceMapper.map_to_all_gpus()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person"],
        attributes={config.category: config.values},
        extra_data=[config.extra_data])
    backend_loader = lambda capsule_files, device: Backend(
        model_bytes=capsule_files["classification_gait_model.pb"],
        metadata_bytes=capsule_files["dataset_metadata.json"],
        model_name="inception_resnet_v2",
        device=device)
    options = {}
