import argparse
from argparse import Namespace

from time import time
from typing import Dict, List

import cv2
import numpy as np
import onnx
import onnxruntime
import torch

from torchvision.transforms import ToTensor
from tqdm import tqdm

from demo_pair import ImageLoader
from nets.aliked import ALIKED

import deform_conv2d_onnx_exporter

deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()


def parse_args():
    parser = argparse.ArgumentParser(description="ALIKED to onnx.")

    # Input data.
    parser.add_argument("input", type=str, default="", help="Image directory.")

    # Model settings.
    parser.add_argument(
        "--model",
        choices=["aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32"],
        default="aliked-n16rot",
        help="The model configuration",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Path for model saving.",
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

    # ONNX export options.
    parser.add_argument(
        "--opset_version",
        type=int,
        # default=11,
        default=16,
        help="Opset version. See ONNX documentation for list of supported "
        "operations in each opset version. Also, opset_version should match "
        "used TensorRT version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set verbose output during exporting to ONNX format.",
    )
    args = parser.parse_args()
    return args


def to_numpy(tensor: torch.Tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def load_model(args: Namespace) -> torch.nn.Module:
    print("Loading model ...")
    model = ALIKED(
        model_name=args.model,
        device=args.device,
        top_k=args.top_k,
        scores_th=args.scores_th,
        n_limit=args.n_limit,
    )
    model.eval()
    print("Model loaded!")
    return model


def load_data(args: Namespace) -> torch.Tensor:
    image_loader = ImageLoader(args.input)
    image = image_loader[0]  # (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = (
        ToTensor()(image).to(args.device).unsqueeze(0)
    )  # (B, C, H, W)
    return image_tensor


def convert_to_onnx(
    args: Namespace, model: torch.nn.Module, image_tensor: torch.Tensor
):
    print("Converting model to ONNX ...")
    export_save_path = args.model_output
    opset_version = args.opset_version
    torch.onnx.export(
        model,
        image_tensor,
        export_save_path,
        export_params=True,
        verbose=args.verbose,
        opset_version=opset_version,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        do_constant_folding=True,
        input_names=[
            "image",
        ],
        output_names=[
            "keypoints",
            "descriptors",
            "scores",
            # "score_dispersity",
            # "score_map",
        ],
        # dynamic_axes={
        #     "image": {1: "num_channels", 2: "height", 3: "width"},
        #     "keypoints": {1: "num_keypoints"},
        #     "descriptors": {1: "num_keypoints"},
        #     "scores": {1: "num_keypoints"},
        #     "score_dispersity": {1: "num_keypoints"},
        #     "score_map": {2: "height", 3: "width"},
        # },
    )
    print(f"Model converted to ONNX and stored to {export_save_path}")
    onnx_model = onnx.load(export_save_path)
    onnx.checker.check_model(
        onnx_model
    )  # This would raise an exception if failed.
    print("Model consistency checked, everything is fine!")
    return export_save_path


def get_onnx_predictions(
    onnx_model_path: str, image_tensor: torch.Tensor
) -> List[np.ndarray]:

    # This fails becausse CUDA exection provider expects CUDA 11. In order to
    # make it work with CUDA 12, follow this:
    # https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x
    # (That is note from this table: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
    # providers = [
    #     (
    #         "CUDAExecutionProvider",
    #         {
    #             "device_id": torch.cuda.current_device(),
    #             "user_compute_stream": str(
    #                 torch.cuda.current_stream().cuda_stream
    #             ),
    #         },
    #     )
    # ]
    # sess_options = onnxruntime.SessionOptions()
    # ort_session = onnxruntime.InferenceSession(
    #     onnx_model_path, sess_options=sess_options, providers=providers
    # )

    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    defined_inputs = ort_session.get_inputs()
    input_data = [image_tensor]
    ort_inputs = {
        defined_inputs[index].name: to_numpy(input_data[index])
        for index in range(len(defined_inputs))
    }

    all_timings = []
    for _ in tqdm(range(20), desc="ONNX timing"):
        start_time = time()
        pred_onnx = ort_session.run(
            [
                "keypoints",
                "descriptors",
                "scores",
                # "score_dispersity",
                # "score_map",
            ],
            ort_inputs,
        )
        end_time = time()
        duration = (end_time - start_time) * 1000
        all_timings.append(duration)
    print(f"timings.max: {np.max(all_timings[3:])}")
    print(f"timings.min: {np.min(all_timings[3:])}")
    print(f"timings.mean: {np.mean(all_timings[3:])}")
    print(f"timings.median: {np.median(all_timings[3:])}")
    return pred_onnx


def compare_outputs(
    pred_onnx: List[np.ndarray], pred_torch: Dict[str, torch.Tensor]
):
    np.testing.assert_allclose(
        pred_onnx[0],
        to_numpy(pred_torch["keypoints"][0]),
        rtol=1e-7,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        pred_onnx[1],
        to_numpy(pred_torch["descriptors"][0]),
        rtol=1e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pred_onnx[2],
        to_numpy(pred_torch["scores"][0]),
        rtol=1e-7,
        atol=5e-5,
    )
    # np.testing.assert_allclose(
    #     pred_onnx[3],
    #     to_numpy(pred_torch["score_dispersity"][0]),
    #     rtol=1e-7,
    #     atol=2e-5,
    # )
    # np.testing.assert_allclose(
    #     pred_onnx[4],
    #     to_numpy(pred_torch["score_map"]),
    #     rtol=1e-7,
    #     atol=5e-5,
    # )

    print("Outputs are the same before and after conversion!")


def main():
    args = parse_args()

    # Load model
    model = load_model(args)

    # Load data.
    image_tensor = load_data(args)

    # do inference
    pred_torch = model(image_tensor)
    # model: aliked-n16rot
    # pred_torch = {
    #     "keypoints": torch.tensor, shape: (B, N, 2),
    #     "descriptors": torch.tensor, shape: (B, N, 128),
    #     "scores": torch.tensor, shape: (B, N,),
    #     "score_dispersity": torch.tensor, shape: (B, N)
    #     "score_map": torch.tensor, shape: (B, 1, image_height, image_width),
    #     "time": float,
    # }

    # convert model
    export_save_path = convert_to_onnx(args, model, image_tensor)

    # do inference after conversion
    pred_onnx = get_onnx_predictions(export_save_path, image_tensor)

    # compare diffs
    compare_outputs(pred_onnx, pred_torch)


if __name__ == "__main__":
    main()
