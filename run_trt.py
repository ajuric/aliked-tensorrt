#!/usr/bin/env python3

import argparse

from time import time

import cv2
import numpy as np

from torchvision.transforms import ToTensor
from tqdm import tqdm

from demo_pair import ImageLoader
from trt_model import LOGGER_DICT, TRTInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing TensorRT ALIKED interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings.
    parser.add_argument(
        "--model_path",
        required=True,
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    trt_logger = LOGGER_DICT["verbose"]
    model = TRTInference(args.model_path, args.model, trt_logger)
    print("TensorRT model loaded!")

    timings = []
    image_loader = ImageLoader(args.images_dir)
    for image in tqdm(image_loader, desc="TRT Inference"):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = ToTensor()(image).unsqueeze(0)  # (B, C, H, W)
        image = image_tensor.numpy()
        # image = np.expand_dims(image.transpose(2, 0, 1), 0) # (1, C, H, W)
        # _, _, h, w = image.shape
        # wh = np.array([w - 1, h - 1])

        start_time = time()
        keypoints, scores, descriptors = model.infer(image)
        end_time = time()
        duration = (end_time - start_time) * 1000  # convert to ms.
        timings.append(duration)
        # keypoints = wh * (keypoints + 1) / 2

    timings = timings[5:]
    print(f"mean: {np.mean(timings):.2f}ms")
    print(f"median: {np.median(timings):.2f}ms")
    print(f"min: {np.min(timings):.2f}ms")
    print(f"max: {np.max(timings):.2f}ms")
