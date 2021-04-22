import numpy as np
import cupy as cp

import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
import time

from typing import Dict, List, Tuple, Optional, Any

from vcap import (
    Crop,
    DetectionNode,
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    BaseBackend,
    rect_to_coords,
)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_engine(trt_runtime, engine_data):
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class AllocatedBuffer:
    def __init__(self, inputs_, outputs_, bindings_, stream_):
        self.inputs = inputs_
        self.outputs = outputs_
        self.bindings = bindings_
        self.stream = stream_


class BaseTensorRTBackend(BaseBackend):
    def __init__(self, engine_bytes, width, height):
        super().__init__()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # load the engine
        self.trt_engine = load_engine(self.trt_runtime, engine_bytes)
        # create execution context
        self.context = self.trt_engine.create_execution_context()
        # create buffers for inference
        self.buffers = {}
        for batch_size in range(1, self.trt_engine.max_batch_size + 1):
            inputs, outputs, bindings, stream = self.allocate_buffers(
                batch_size=batch_size)
            self.buffers[batch_size] = AllocatedBuffer(inputs, outputs, bindings,
                                                       stream)

        self.engine_width = width
        self.engine_height = height

        self._prepare_post_process()

    def batch_predict(self, input_data_list: List[Any]) -> List[Any]:
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

    def _process_batch(self, input_data: List[np.array]):
        pre_batch_time = time.time()
        batch_size = len(input_data)
        # input_data_cuppy = [cp.array(data) for data in input_data]
        """Ideas
        1) Try raveling before concatenate
        2) Instead of concatenating, try generating an array of the size and shape of what the concatenated image WOULD 
            be, then copy the raveled images into their respective places in the array
        """
        ravel_time = time.time()
        raveled_input = [data.ravel() for data in input_data]
        print("batch_size:", batch_size, "ravel time:", int(round((time.time() - ravel_time) * 1000)),
              int(round((time.time() - ravel_time) * 1000)) / batch_size)

        concatenate_time = time.time()
        batched_image = np.concatenate(raveled_input, axis=0)
        #image_size = len(raveled_input[0])
        #batched_image = np.zeros((1, batch_size * image_size))
        #for index, image in enumerate(batched_image):
        #    batched_image[index * image_size:(index + 1) * image_size] = image
        print("batch_size:", batch_size, "concatenate time:", int(round((time.time() - concatenate_time) * 1000)),
              int(round((time.time() - concatenate_time) * 1000)) / batch_size)

        # image_size = self.engine_height * self.engine_width
        # batched_image = np.zeros((1, batch_size * image_size))
        # for index, image in enumerate(batched_image):
        #    for row_index, row in enumerate(image):
        #        batched_image[index * image_size + row_index * self.engine_width:(index+1) * image_size + row_index * self.engine_width] = row

        # for data in input_data:
        #    batch_image_array.append(data.ravel())

        # batched_image = np.concatenate(input_data, axis=0)
        # print(type(batched_image))
        prepared_buffer = self.buffers[batch_size]
        inputs = prepared_buffer.inputs
        outputs = prepared_buffer.outputs
        bindings = prepared_buffer.bindings
        stream = prepared_buffer.stream
        copy_time = time.time()
        # raveled_image = batched_image.ravel()

        # np.copyto(inputs[0].host, raveled_image)
        np.copyto(inputs[0].host, batched_image)
        print("batch_size:", batch_size, "copy time:", int(round((time.time() - copy_time) * 1000)),
              int(round((time.time() - copy_time) * 1000)) / batch_size)

        print("batch_size:", batch_size, "pre_batch_time:", int(round((time.time() - pre_batch_time) * 1000)),
              int(round((time.time() - pre_batch_time) * 1000)) / batch_size)
        detections = self.do_inference(
            bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,
            batch_size=batch_size
        )
        return detections

    def process_frame(self, frame: np.ndarray, detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        pass

    def prepare_inputs(self, frame: np.ndarray, transpose: bool, normalize: bool,
                       mean_subtraction: Optional[Tuple] = None) -> \
            Tuple[np.array, Resize]:
        pre_process_start_time = time.time()
        # h, w, c = frame.shape
        # print(h, w, self.engine_height, self.engine_width)
        resize = Resize(frame).resize(self.engine_width, self.engine_height,
                                      Resize.ResizeType.EXACT)
        # print("resize take:", int(round((time.time() - pre_process_start_time) * 1000)))

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
        # print("prepare input take:", int(round((time.time() - pre_process_start_time) * 1000)))
        return resize.frame, resize

    def allocate_buffers(self, batch_size=1):
        """Allocates host and device buffer for TRT engine inference.
        This function is similair to the one in common.py, but
        converts network outputs (which are np.float32) appropriately
        before writing them to Python buffer. This is needed, since
        TensorRT plugins doesn't support output type description, and
        in our particular case, we use NMS plugin as network output.
        Args:
            engine (trt.ICudaEngine): TensorRT engine
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

    def do_inference(self, bindings, inputs, outputs, stream, batch_size=1):
        inference_start_time = time.time()

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        # todo: try execute synchronously
        self.context.execute(
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
        final_outputs = []
        for i in range(len(batch_outputs[0])):
            final_output = []
            for batch_output in batch_outputs:
                final_output.append(batch_output[i])
            final_outputs.append(final_output)
        print("batch_size:", batch_size,
              "TensorRT inference time: {} ms".format(
                  int(round((time.time() - inference_start_time) * 1000))
              )
              )
        return final_outputs

    def _prepare_post_process(self):
        self.stride = 16
        self.box_norm = 35.0
        self.grid_h = int(self.engine_height / self.stride)
        self.grid_w = int(self.engine_width / self.stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []

        for i in range(self.grid_h):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)

    def _apply_box_norm(self, o1, o2, o3, o4, x, y):
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
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm
        return o1, o2, o3, o4

    def postprocess(self, outputs, min_confidence, analysis_classes, wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """
        # print(len(outputs))
        bbs = []
        class_ids = []
        scores = []
        for c in analysis_classes:

            x1_idx = c * 4 * self.grid_size
            y1_idx = x1_idx + self.grid_size
            x2_idx = y1_idx + self.grid_size
            y2_idx = x2_idx + self.grid_size

            boxes = outputs[0]
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    i = w + h * self.grid_w
                    score = outputs[1][c * self.grid_size + i]
                    if score >= min_confidence:
                        o1 = boxes[x1_idx + w + h * self.grid_w]
                        o2 = boxes[y1_idx + w + h * self.grid_w]
                        o3 = boxes[x2_idx + w + h * self.grid_w]
                        o4 = boxes[y2_idx + w + h * self.grid_w]

                        o1, o2, o3, o4 = self._apply_box_norm(o1, o2, o3, o4, w, h)

                        xmin = int(o1)
                        ymin = int(o2)
                        xmax = int(o3)
                        ymax = int(o4)
                        if wh_format:
                            bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        else:
                            bbs.append([xmin, ymin, xmax, ymax])
                        class_ids.append(c)
                        scores.append(float(score))

        return bbs, class_ids, scores
