#!/usr/bin/env python3

import argparse
import os

from argparse import Namespace

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from aliked_service import (
    PyTorchAlikedService,
    AlikedService,
    TensorRTAlikedService,
)
from demo_pair import ImageLoader
from gpu_utils import show_memory_gpu_usage
from nets.aliked import ALIKED
from trt_model import LOGGER_DICT, TRTInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing TensorRT ALIKED interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O settings.
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory with images.",
    )
    parser.add_argument(
        "--trt_model_path",
        default=None,
        type=str,
        help="Path to model in TRT format. This cannot be used in combination "
        "with --compile argument since torch.compile() is used for PyTorch "
        "models, not for TensorRT.",
    )
    # Model settings.
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mode for PyTorch model.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() optimization for PyTorch model. This cannot "
        "be used in combination with --trt_model_path argument.",
    )
    parser.add_argument(
        "--model",
        choices=["aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32"],
        default="aliked-n16rot",
        help="The model configuration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Running device (default: cuda).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Detect top K keypoints. -1 for threshold based mode, >0 for top "
        "K mode. (default: -1)",
    )
    parser.add_argument(
        "--scores_th",
        type=float,
        default=0.2,
        help="Detector score threshold (default: 0.2).",
    )
    parser.add_argument(
        "--n_limit",
        type=int,
        default=5000,
        help="Maximum number of keypoints to be detected (default: 5000).",
    )

    args = parser.parse_args()

    if args.trt_model_path is not None and args.compile:
        raise ValueError(
            "Both arguments --trt_model_path and --compile are provided! "
        )

    return args


@torch.no_grad()
def compare_fp32_fp16(
    aliked_service_fp32: AlikedService,
    aliked_service_fp16: AlikedService,
    image_loader: ImageLoader,
) -> None:

    for image in tqdm(image_loader, desc="Inference"):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_fp32 = aliked_service_fp32.prepare_data(image)
        keypoints_fp32, descriptors_fp32, scores_fp32 = (
            aliked_service_fp32.infer(image_fp32)
        )
        keypoints_fp32 = keypoints_fp32[0].cpu().numpy()
        scores_fp32 = scores_fp32[0].cpu().numpy()
        descriptors_fp32 = descriptors_fp32[0].cpu().numpy()
        _, _, h, w = image_fp32.shape
        wh_fp32 = np.array([w - 1, h - 1], dtype=np.float32)
        keypoints_fp32 = wh_fp32 * (keypoints_fp32 + 1) / 2

        image_fp16 = aliked_service_fp16.prepare_data(image)
        keypoints_fp16, descriptors_fp16, scores_fp16 = (
            aliked_service_fp16.infer(image_fp16)
        )
        # keypoints_fp16 = keypoints_fp16[
        #     0
        # ].float()  # Conversion to fp32 is intended!
        # # Because this is output, and we only care that model weights and
        # # computation is fp16.
        keypoints_fp16 = keypoints_fp16[0].cpu().numpy()
        scores_fp16 = scores_fp16[0].cpu().numpy()
        descriptors_fp16 = descriptors_fp16[0].cpu().numpy()
        # keypoints_fp16 = keypoints_fp16.reshape((-1, 2))
        # scores_fp16 = scores_fp16
        # descriptors_fp16 = descriptors_fp16.reshape(
        #     (-1, aliked_service_fp16._model.dim)
        # )
        _, _, h, w = image_fp16.shape
        wh_fp16 = np.array([w - 1, h - 1], dtype=np.float32)
        keypoints_fp16 = wh_fp16 * (keypoints_fp16 + 1) / 2

        # indices_fp32 = np.argsort(keypoints_fp32[:, 0], axis=0)
        # keypoints_fp32 = keypoints_fp32[indices_fp32]
        # scores_fp32 = scores_fp32[indices_fp32]
        # descriptors_fp32 = descriptors_fp32[indices_fp32]

        # indices_fp16 = np.argsort(keypoints_fp16[:, 0], axis=0)
        # keypoints_fp16 = keypoints_fp16[indices_fp16]
        # scores_fp16 = scores_fp16[indices_fp16]
        # descriptors_fp16 = descriptors_fp16[indices_fp16]

        # vidjeti li je soriranje sada dobro?
        # probati s manje keypointova, tipa 20?

        mutual_indices1, mutual_indices2 = find_mutual_closest_keypoints(
            keypoints_fp32, keypoints_fp16
        )

        keypoints_fp32 = keypoints_fp32[mutual_indices1]
        keypoints_fp16 = keypoints_fp16[mutual_indices2]
        scores_fp32 = scores_fp32[mutual_indices1]
        scores_fp16 = scores_fp16[mutual_indices2]
        descriptors_fp32 = descriptors_fp32[mutual_indices1]
        descriptors_fp16 = descriptors_fp16[mutual_indices2]

        plot_keypoints(
            image,
            keypoints_fp32,
            keypoints_fp16,
        )

        np.testing.assert_allclose(
            keypoints_fp32,
            keypoints_fp16,
            rtol=5e-3,
            atol=1e-3,
            verbose=True,
        )
        np.testing.assert_allclose(
            scores_fp32, scores_fp16, rtol=5e-2, atol=5e-2, verbose=True
        )
        descriptors_similarity = np.einsum(
            "ij,ij->i", descriptors_fp32, descriptors_fp16
        )
        ones = np.ones_like(descriptors_similarity)
        np.testing.assert_allclose(
            descriptors_similarity, ones, rtol=1e-2, atol=1e-2
        )


def plot_keypoints(image, keypoints_fp32, keypoints_fp16):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.scatter(keypoints_fp32[:, 0], keypoints_fp32[:, 1], c="r", s=0.5)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    plt.scatter(keypoints_fp16[:, 0], keypoints_fp16[:, 1], c="b", s=0.5)
    plt.show()
    plt.close()


# Method which given 2 sets of keypoints finds pairs of all closest points
# which are mutualy closest.
def find_mutual_closest_keypoints(keypoints1, keypoints2):
    # Find all pairs of closest keypoints
    distances = np.sqrt(
        np.sum((keypoints1[:, None, :] - keypoints2[None, :, :]) ** 2, axis=2)
    )
    # indices1, indices2 = np.unravel_index(
    #     np.argmin(distances, axis=1), distances.shape
    # )
    indices1 = np.argmin(distances, axis=1)
    indices2 = np.argmin(distances, axis=0)

    # Find all pairs of mutual closest points
    mutual_indices1 = []
    mutual_indices2 = []
    for index in range(indices1.shape[0]):
        if index == indices2[indices1[index]]:
            mutual_indices1.append(index)
            mutual_indices2.append(indices1[index])

    return mutual_indices1, mutual_indices2


def create_aliked_service(args: Namespace) -> AlikedService:
    if args.trt_model_path is None:  # Use PyTorch version.
        model = ALIKED(
            model_name=args.model,
            device=args.device,
            top_k=args.top_k,
            scores_th=args.scores_th,
            n_limit=args.n_limit,
        )
        if args.mode == "fp16":
            model.half()
        if args.compile:
            model = model.to("cuda")
            model = torch.jit.trace(
                model,
                torch.randn(
                    1, 3, 480, 640, device="cuda", dtype=torch.float32
                ),
            )
            model = torch.compile(model, mode="reduce-overhead")
        aliked_service = PyTorchAlikedService(model, mode=args.mode)
    else:  # Use TRT version.
        trt_logger = LOGGER_DICT["verbose"]
        model = TRTInference(args.trt_model_path, args.model, trt_logger)
        aliked_service = TensorRTAlikedService(model)
    print(f"Loaded {aliked_service.name} service.")
    return aliked_service


def main():
    args = parse_args()

    args.mode = "fp32"
    trt_model_path = args.trt_model_path
    args.trt_model_path = None
    aliked_service_fp32 = create_aliked_service(args)
    args.mode = "amp"
    # args.trt_model_path = trt_model_path
    aliked_service_fp16 = create_aliked_service(args)

    image_loader = ImageLoader(args.images_dir)

    warmup_image = image_loader[0]
    warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
    aliked_service_fp32.warmup(warmup_image)
    aliked_service_fp16.warmup(warmup_image)

    show_memory_gpu_usage()
    compare_fp32_fp16(
        aliked_service_fp32=aliked_service_fp32,
        aliked_service_fp16=aliked_service_fp16,
        image_loader=image_loader,
    )
    show_memory_gpu_usage()


if __name__ == "__main__":
    main()
