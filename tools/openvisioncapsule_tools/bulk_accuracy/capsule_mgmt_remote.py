from brainframe.api import BrainFrameAPI

from tools.bulk_accuracy.capsule_mgmt_basic import BasicCapsuleManagement


class RemoteCapsuleManagement(BasicCapsuleManagement):
    def __init__(self, capsule_names, options=None):
        self.capsule_names = capsule_names
        self.options = options
        self.previous_capsule_options = {}
        self.api = None
        self.initial_global_capsule_options()

    def initial_global_capsule_options(self):
        bf_server_url = "http://localhost:80"
        self.api: BrainFrameAPI = BrainFrameAPI(bf_server_url)
        print("Connecting to Brainframe Server...")
        self.api.wait_for_server_initialization()
        if self.options is not None:
            for capsule_name in self.capsule_names:
                current_options = self.api.get_capsule_option_vals(
                    capsule_name=capsule_name
                )
                self.previous_capsule_options[capsule_name] = current_options
                self.api.set_capsule_option_vals(
                    capsule_name=capsule_name, option_vals=self.options
                )

    def recover_global_capsule_options(self):
        for capsule_name in self.previous_capsule_options:
            previous_options = self.previous_capsule_options[capsule_name]
            self.api.set_capsule_option_vals(
                capsule_name=capsule_name, option_vals=previous_options
            )

    def process_image(self, frame, detections=None):
        detections = self.api.process_image(frame, self.capsule_names, {})
        return detections

    def get_positions(self, bbox):
        p1, p2 = bbox[0], bbox[2]
        x1, x2 = p1[0], p2[0]
        y1, y2 = p1[1], p2[1]
        return x1, x2, y1, y2
