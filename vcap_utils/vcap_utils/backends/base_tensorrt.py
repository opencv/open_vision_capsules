import numpy as np

import pycuda.driver as cuda
import tensorrt as trt

from typing import Dict, List, Tuple, Optional, Generator

from vcap import (
    Resize,
    BaseBackend,
    rect_to_coords,
    DetectionNode,
)
from vcap_utils.algorithms import non_max_suppression


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class AllocatedBuffer:
    def __init__(self, inputs_, outputs_, bindings_, stream_):
        self.inputs = inputs_
        self.outputs = outputs_
        self.bindings = bindings_
        self.stream = stream_


class BaseTensorRTBackend(BaseBackend):
    def __init__(self, engine_bytes: bytes, width: int, height: int, device_name: str):
        super().__init__()
        gpu_devide_id = int(device_name[4:])
        cuda.init()
        dev = cuda.Device(gpu_devide_id)
        self.cuda_context = dev.make_context()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # load the engine
        self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_bytes)
        # create execution context
        self.trt_context = self.trt_engine.create_execution_context()
        # create buffers for inference
        self.buffers = {}
        for batch_size in range(1, self.trt_engine.max_batch_size + 1):
            inputs, outputs, bindings, stream = self.allocate_buffers(
                batch_size=batch_size)
            self.buffers[batch_size] = AllocatedBuffer(inputs, outputs, bindings,
                                                       stream)

        self.engine_width = width
        self.engine_height = height

        # preallocate resources for post process
        # todo: post process is only need for detectors
        self._prepare_post_process()

    def batch_predict(self, input_data_list: List[np.ndarray]) \
            -> Generator[List[DetectionNode], None, None]:
        task_size = len(input_data_list)
        curr_index = 0
        while curr_index < task_size:
            if curr_index + self.trt_engine.max_batch_size <= task_size:
                end_index = curr_index + self.trt_engine.max_batch_size
            else:
                end_index = task_size
            batch = input_data_list[curr_index:end_index]
            curr_index = end_index
            for result in self._process_batch(batch):
                yield result

    def _process_batch(self, input_data: List[np.array]) -> List[List[float]]:
        batch_size = len(input_data)
        prepared_buffer = self.buffers[batch_size]
        inputs = prepared_buffer.inputs
        # todo: get dtype from engine
        inputs[0].host = np.ascontiguousarray(input_data, dtype=np.float32)

        return self._do_inference(
            bindings=prepared_buffer.bindings,
            inputs=inputs,
            outputs=prepared_buffer.outputs,
            stream=prepared_buffer.stream,
            batch_size=batch_size
        )

    def prepare_inputs(self, frame: np.ndarray, transpose: bool, normalize: bool,
                       mean_subtraction: Optional[Tuple] = None) -> \
            Tuple[np.array, Resize]:
        resize = Resize(frame).resize(self.engine_width, self.engine_height,
                                      Resize.ResizeType.EXACT)
        if transpose:
            resize.frame = np.transpose(resize.frame, (2, 0, 1))
        if normalize:
            resize.frame = (1.0 / 255.0) * resize.frame
        if mean_subtraction is not None:
            if len(mean_subtraction) != 3:
                raise RuntimeError("Invalid mean subtraction")
            resize.frame = resize.frame.astype("float64")
            resize.frame[..., 0] -= mean_subtraction[0]
            resize.frame[..., 1] -= mean_subtraction[1]
            resize.frame[..., 2] -= mean_subtraction[2]
        return resize.frame, resize

    def allocate_buffers(self, batch_size: int = 1) -> \
            Tuple[List[HostDeviceMem], List[HostDeviceMem], List[int], cuda.Stream]:
        """Allocates host and device buffer for TRT engine inference.
        Args:
            batch_size: batch size for the input/output memory
        Returns:
            inputs [HostDeviceMem]: engine input memory
            outputs [HostDeviceMem]: engine output memory
            bindings [int]: buffer to device bindings
            stream (cuda.Stream): cuda stream for engine inference synchronization
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.trt_engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def _do_inference(self, bindings: List[int],
                      inputs: List[HostDeviceMem],
                      outputs: List[HostDeviceMem],
                      stream: cuda.Stream,
                      batch_size: int = 1) -> List[List[float]]:
        # Transfer input data to the GPU.
        self.cuda_context.push()
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        # todo: use async or sync api?
        # According to https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html#optimize-python
        # the performance should be almost identical
        self.trt_context.execute(
            batch_size=batch_size, bindings=bindings
        )
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        batch_outputs = []
        for out in outputs:
            entire_out_array = np.array(out.host)
            out_array_by_batch = np.split(entire_out_array, batch_size)
            out_lists = [out_array.tolist() for out_array in out_array_by_batch]
            batch_outputs.append(out_lists)
        final_outputs = list(zip(*batch_outputs))
        final_outputs = [list(item) for item in final_outputs]
        self.cuda_context.pop()
        return final_outputs

    def _prepare_post_process(self):
        stride = 16
        self.box_norm = 35.0
        self.grid_h = int(self.engine_height / stride)
        self.grid_w = int(self.engine_width / stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_h = [(i * stride + 0.5) / self.box_norm for i in range(self.grid_h)]
        self.grid_centers_w = [(i * stride + 0.5) / self.box_norm for i in range(self.grid_w)]

    def _apply_box_norm(self, o1: float, o2: float, o3: float, o4: float, x: int, y: int) -> \
            Tuple[int, int, int, int]:
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            int: rescaled first argument
            int: rescaled second argument
            int: rescaled third argument
            int: rescaled fourth argument
        """
        xmin = int((o1 - self.grid_centers_w[x]) * -self.box_norm)
        ymin = int((o2 - self.grid_centers_h[y]) * -self.box_norm)
        xmax = int((o3 + self.grid_centers_w[x]) * self.box_norm)
        ymax = int((o4 + self.grid_centers_h[y]) * self.box_norm)
        return xmin, ymin, xmax, ymax

    def parse_detection_results(
            self, results: List[List[float]],
            resize: Resize,
            label_map: Dict[int, str],
            min_confidence: float = 0.0,
    ) -> List[DetectionNode]:
        detection_nodes: List[DetectionNode] = []
        for class_id, class_name in label_map.items():
            x1_idx = class_id * 4 * self.grid_size
            y1_idx = x1_idx + self.grid_size
            x2_idx = y1_idx + self.grid_size
            y2_idx = x2_idx + self.grid_size

            boxes = results[0]
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    i = w + h * self.grid_w
                    score = results[1][class_id * self.grid_size + i]
                    if score >= min_confidence:
                        o1 = boxes[x1_idx + w + h * self.grid_w]
                        o2 = boxes[y1_idx + w + h * self.grid_w]
                        o3 = boxes[x2_idx + w + h * self.grid_w]
                        o4 = boxes[y2_idx + w + h * self.grid_w]
                        xmin, ymin, xmax, ymax = self._apply_box_norm(o1, o2, o3, o4, w, h)
                        detection_nodes.append(DetectionNode(
                            name=class_name,
                            coords=rect_to_coords(
                                [xmin, ymin, xmax, ymax]
                            ),
                            extra_data={"detection_confidence": score},
                        ))
        detection_nodes = non_max_suppression(detection_nodes, max_bbox_overlap=0.5)
        resize.scale_and_offset_detection_nodes(detection_nodes)
        return detection_nodes

    def close(self) -> None:
        super().close()
        self.cuda_context.pop()
