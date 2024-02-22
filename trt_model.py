
from typing import Any
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from torchvision.transforms import ToTensor
from nets.aliked import ALIKED_CFGS


LOGGER_DICT = {
    "warning": trt.Logger(trt.Logger.WARNING),
    "info": trt.Logger(trt.Logger.INFO),
    "verbose": trt.Logger(trt.Logger.VERBOSE),
}


class TRTInference:
    def __init__(
        self,
        trt_engine_path: str,
        model_type: str,
        trt_logger: trt.Logger,
    ):
        self.trt_logger = trt_logger

        # get configurations
        _, _, _, _, self.dim, _, _ = [
            v for _, v in ALIKED_CFGS[model_type].items()
        ]

        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        trt.init_libnvinfer_plugins(self.trt_logger, "")
        runtime = trt.Runtime(self.trt_logger)

        # deserialize engine
        with open(trt_engine_path, "rb") as trt_engine_file:
            buf = trt_engine_file.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        tensor_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
        ]
        for tensor_name in tensor_names:
            msg = "\n=============================="
            data_type = (
                np.float32
                if engine.get_tensor_dtype(tensor_name) == trt.DataType.FLOAT
                else np.int32
            )
            msg += f"\n{tensor_name}: {data_type}"
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            msg += (
                "\nengine.get_tensor_shape(tensor_name): "
                f"{engine.get_tensor_shape(tensor_name)}"
            )
            msg += f"\nsize: {size}"
            msg += "\n=============================="
            self.trt_logger.log(trt.Logger.INFO, msg)
            host_mem = cuda.pagelocked_empty(size, data_type)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:  # == trt.TensorIOMode.OUTPUT
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
    
    def warmup(self, image: np.ndarray, num_iterations: int = 3) -> None:
        print("Starting warm-up ...")
        for _ in range(num_iterations):
            self.run(image)
        print("Warm-up done!")
    
    def run(self, image):
        image_tensor = ToTensor()(image).unsqueeze(0)  # (B, C, H, W)
        image = image_tensor.numpy()
        # image = np.expand_dims(image.transpose(2, 0, 1), 0) # (1, C, H, W)
        _, _, h, w = image.shape
        wh = np.array([w - 1, h - 1])
        keypoints, scores, descriptors = self.infer(image)
        keypoints = keypoints.reshape(-1, 2)
        keypoints = wh * (keypoints + 1) / 2

        return {
            "keypoints": keypoints.reshape(-1, 2),  # N 2
            "descriptors": descriptors.reshape(-1, self.dim),  # N D
            "scores": scores,  # B N D
        }

    def infer(self, image):
        self.cfx.push()

        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # Copy data to GPU.
        for index, host_input in enumerate(host_inputs):
            pagelocked_buffer = host_input
            flattened_data = image[index].flatten()
            data_size = flattened_data.shape[0]
            np.copyto(pagelocked_buffer[:data_size], flattened_data)

        for cuda_inp, host_inp in zip(cuda_inputs, host_inputs):
            cuda.memcpy_htod_async(cuda_inp, host_inp, stream)

        # Inference.
        context.execute_async_v2(
            bindings=bindings, stream_handle=stream.handle
        )
        # context.execute_async_v3(
        #     # bindings=bindings, stream_handle=stream.handle
        #     stream_handle=stream.handle
        # )

        # Copy to host.
        for cuda_out, host_out in zip(cuda_outputs, host_outputs):
            cuda.memcpy_dtoh_async(host_out, cuda_out, stream)
        stream.synchronize()

        keypoints, scores, descriptors = host_outputs

        self.cfx.pop()
        return keypoints, scores, descriptors

    def __del__(self):
        self.cfx.pop()
