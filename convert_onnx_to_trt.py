import argparse
from argparse import Namespace
import time

import tensorrt as trt


LOGGER_DICT = {
    "warning": trt.Logger(trt.Logger.WARNING),
    "info": trt.Logger(trt.Logger.INFO),
    "verbose": trt.Logger(trt.Logger.VERBOSE),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for converting into ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths.
    parser.add_argument(
        "--model_onnx_path",
        type=str,
        required=True,
        help="Path to model saved in ONNX format.",
    )
    parser.add_argument(
        "--model_trt_path",
        type=str,
        required=True,
        help="Path for saving model in TensorRT format.",
    )

    # TensorRT conversion options.
    parser.add_argument(
        "--optimization_level",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=5,
        help="Builder optimization level. Greater level should provide better "
        "converion results, since optimization algorithm tries more tactics "
        "during conversion. Maximum is 5.",
    )
    parser.add_argument(
        "--precision_mode",
        type=str,
        choices=["FP16", "FP32"],
        default="FP32",
        help="Precision mode used in converted model.",
    )
    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=2048,
        help="Max workspace size in MiB.",
    )

    args, _ = parser.parse_known_args()
    return args


def create_profile_with_shapes(builder: trt.Builder):
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "image",
        (1, 3, 376, 1241),
        (1, 3, 376, 1241),
        (1, 3, 376, 1241),
    )
    return profile


def prepare_config(
    args: Namespace, config: trt.IBuilderConfig
) -> trt.IBuilderConfig:
    config.max_workspace_size = args.max_workspace_size * (1024**2)
    if args.precision_mode == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = args.optimization_level
    return config


def build_engine(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
) -> trt.IHostMemory:
    print("Building engine. This might take a while .........")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise ValueError("Building TensorRT Engine failed.")
    end_time = time.time()
    duration = end_time - start_time
    print(f"Engine built, time taken: {duration:.2f}s.")
    return serialized_engine


def build_tensorrt_network(
    args: Namespace,
    trt_logger: trt.Logger,
    explicit_batch: int,
):
    with trt.Builder(
        trt_logger
    ) as builder, builder.create_builder_config() as config:
        profile = create_profile_with_shapes(builder)
        # config.add_optimization_profile(profile)

        with builder.create_network(explicit_batch) as network, trt.OnnxParser(
            network, trt_logger
        ) as parser:
            with open(args.model_onnx_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            print(
                "INFO: Batch size is set to 1. If this needs to be changed, "
                "code needs to be upadted."
            )
            batch_size = 1
            builder.max_batch_size = batch_size

            config = prepare_config(args, config)
            serialized_engine = build_engine(builder, network, config)

            with open(args.model_trt_path, "wb") as output_file:
                output_file.write(serialized_engine)


def main():
    args = parse_args()
    trt_logger = LOGGER_DICT["verbose"]
    explicit_batch = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )

    build_tensorrt_network(args, trt_logger, explicit_batch)


if __name__ == "__main__":
    main()
