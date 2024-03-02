from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from torch.nn import Module
from torch import Tensor
from torchvision.transforms import ToTensor

from trt_model import TRTInference


class AlikedService(ABC):

    def __init__(self, name: str, model) -> None:
        self._name = name
        self._model = model

    @property
    def name(self):
        return self._name

    @abstractmethod
    def prepare_data(self, image: np.ndarray) -> np.ndarray | Tensor:
        pass

    def warmup(self, image: np.ndarray, num_iterations: int = 3) -> None:
        print("Starting warm-up ...")
        image = self.prepare_data(image)
        for _ in range(num_iterations):
            self.infer(image)
        print("Warm-up done!")

    @abstractmethod
    def infer(
        self, image: np.ndarray | Tensor
    ) -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        | Tuple[Tensor, Tensor, Tensor]
    ):
        pass


class PyTorchAlikedService(AlikedService):

    def __init__(self, model: Module, mode: str = "fp32") -> None:
        super().__init__("PyTorch-ALIKED", model)
        self._mode = mode

    def prepare_data(self, image: np.ndarray) -> Tensor:
        img_tensor = ToTensor()(image)
        # img_tensor = img_tensor.to(self._model.device).unsqueeze_(0)
        img_tensor = img_tensor.to("cuda").unsqueeze_(0)
        if self._mode == "fp16":
            img_tensor = img_tensor.half()
        return img_tensor

    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # predictions = self._model.forward(image)
        # return (
        #     predictions["keypoints"],
        #     predictions["descriptors"],
        #     predictions["scores"],
        # )
        if self._mode == "amp":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                keypoints, descriptors, scores = self._model.forward(image)
        else:
            keypoints, descriptors, scores = self._model.forward(image)
        return keypoints, descriptors, scores


class TensorRTAlikedService(AlikedService):

    def __init__(self, model: TRTInference) -> None:
        super().__init__("TenosrRT-ALIKED", model)

    def prepare_data(self, image: np.ndarray) -> np.ndarray:
        image_tensor = ToTensor()(image).unsqueeze(0)  # (B, C, H, W)
        image = image_tensor.numpy()
        return image

    def infer(
        self, image: np.ndarray | Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._model.infer(image)
