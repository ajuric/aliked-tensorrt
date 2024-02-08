import argparse

from time import time

import cv2
import numpy as np

from torchvision.transforms import ToTensor
from tqdm import tqdm

from demo_pair import ImageLoader
from nets.aliked import ALIKED


def parse_args():
    parser = argparse.ArgumentParser(description="ALIKED image pair Demo.")
    parser.add_argument("input", type=str, default="", help="Image directory.")
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
    return args


def main():
    args = parse_args()

    model = ALIKED(
        model_name=args.model,
        device=args.device,
        top_k=args.top_k,
        scores_th=args.scores_th,
        n_limit=args.n_limit,
    )

    timings = []
    image_loader = ImageLoader(args.input)
    for image in tqdm(image_loader, desc="PyTorch Inference"):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = ToTensor()(image).unsqueeze(0).to(args.device)  # (B, C, H, W)

        start_time = time()
        keypoints, scores, descriptors = model.forward(image_tensor)
        end_time = time()
        duration = (end_time - start_time) * 1000  # convert to ms.
        timings.append(duration)
        # keypoints = wh * (keypoints + 1) / 2

    timings = timings[5:]
    print(f"mean: {np.mean(timings):.2f}ms")
    print(f"median: {np.median(timings):.2f}ms")
    print(f"min: {np.min(timings):.2f}ms")
    print(f"max: {np.max(timings):.2f}ms")


if __name__ == "__main__":
    main()
