from typing import Dict

from vcap import BaseCapsule, NodeDescription, DeviceMapper, BaseBackend

from . import config
from .backend import Backend


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
    options = {}

    @staticmethod
    def backend_loader(capsule_files: Dict[str, bytes], device: str) \
            -> BaseBackend:

        # Real capsules do not need to do this check. This is only to provide
        # a warning for this example because the model is not included in the
        # repo.
        model_filename = "classification_gait_model.pb"
        try:
            model_file = capsule_files[model_filename]
        except KeyError as exc:
            message = f"Model [{model_filename}] not found. Did you make " \
                      f"sure to run tests? Example models files are not " \
                      f"stored directly in the repo, but are downloaded " \
                      f"when tests are run."
            raise FileNotFoundError(message) from exc

        return Backend(model_bytes=model_file,
                       metadata_bytes=capsule_files["dataset_metadata.json"],
                       model_name="inception_resnet_v2",
                       device=device)
