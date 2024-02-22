from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

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

    def warmup(self, image: np.ndarray) -> None:
        self._model.warmup(image)

    @abstractmethod
    def infer(
        self, image: np.ndarray | Tensor
    ) -> (
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        | Tuple[Tensor, Tensor, Tensor]
    ):
        pass


class PyTorchAlikedService(AlikedService):

    def __init__(self, model: Module) -> None:
        super().__init__("PyTorch-ALIKED", model)

    def prepare_data(self, image: np.ndarray) -> Tensor:
        img_tensor = ToTensor()(image)
        img_tensor = img_tensor.to(self._model.device).unsqueeze_(0)
        return img_tensor

    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predictions = self._model.forward(image)
        return (
            predictions["keypoints"],
            predictions["descriptors"],
            predictions["scores"],
        )


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
