#!/usr/bin/env python3

import argparse
import os

from argparse import Namespace
from time import time

import cv2
import numpy as np

from tqdm import tqdm

from aliked_service import (
    TensorRTAlikedService,
    PyTorchAlikedService,
    AlikedService,
)
from demo_pair import ImageLoader
from nets.aliked import ALIKED
from trt_model import LOGGER_DICT, TRTInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing TensorRT ALIKED interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings.
    parser.add_argument(
        "--trt_model_path",
        default=None,
        type=str,
        help="Path to model in TRT format.",
    )
    parser.add_argument(
        "--model",
        choices=["aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32"],
        default="aliked-n16rot",
        help="The model configuration",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory with images.",
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
    return args


def get_gpu_memory_usage() -> int:
    _NVIDIA_SMI_COMMAND = (
        "nvidia-smi "
        "--query-compute-apps=pid,used_memory "
        "--format=csv,noheader,nounits"
    )
    # If multiple GPU-processes are running, we get multiple results, splitted
    # with new-line char.
    results = os.popen(_NVIDIA_SMI_COMMAND).read().strip().split("\n")

    current_pid = os.getpid()
    for result in results:
        pid, used_memory = result.split(",")
        used_memory = int(used_memory)
        if int(pid) == current_pid: # This is our process!
            return used_memory

    return -1  # Indicate process not found.


def measure(aliked_service: AlikedService, image_loader: ImageLoader) -> None:
    timings = []
    for image in tqdm(image_loader, desc="Inference"):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = aliked_service.prepare_data(image)

        start_time = time()
        keypoints, scores, descriptors = aliked_service.infer(image)
        end_time = time()
        duration = (end_time - start_time) * 1000  # convert to ms.
        timings.append(duration)

    print(f"mean: {np.mean(timings):.2f}ms")
    print(f"median: {np.median(timings):.2f}ms")
    print(f"min: {np.min(timings):.2f}ms")
    print(f"max: {np.max(timings):.2f}ms")


def create_aliked_service(args: Namespace):
    if args.trt_model_path is None:  # Use PyTorch version.
        model = ALIKED(
            model_name=args.model,
            device=args.device,
            top_k=args.top_k,
            scores_th=args.scores_th,
            n_limit=args.n_limit,
        )
        aliked_service = PyTorchAlikedService(model)
    else:  # Use TRT version.
        trt_logger = LOGGER_DICT["verbose"]
        model = TRTInference(args.trt_model_path, args.model, trt_logger)
        aliked_service = TensorRTAlikedService(model)
    print(f"Loaded {aliked_service.name} service.")
    return aliked_service


def main():
    args = parse_args()

    aliked_service = create_aliked_service(args)
    image_loader = ImageLoader(args.images_dir)

    warmup_image = image_loader[0]
    warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
    aliked_service.warmup(warmup_image)

    measure(aliked_service=aliked_service, image_loader=image_loader)

    gpu_memory_usage = get_gpu_memory_usage()
    if gpu_memory_usage != -1:
        print(f"GPU memory usage: {gpu_memory_usage} MiB")
    else:
        print("Error: Couldn't find current process in nvidia-smi output.")


if __name__ == "__main__":
    main()
