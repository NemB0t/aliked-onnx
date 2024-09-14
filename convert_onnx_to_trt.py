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

    args, _ = parser.parse_known_args()
    return args


def create_profile_with_shapes(builder: trt.Builder):
    profile = builder.create_optimization_profile()
    # modifying for EuroC dataset
    profile.set_shape(
        "image",
        (1, 3, 480, 752),
        (1, 3, 480, 752),
        (1, 3, 480, 752),
    )
    # profile.set_shape(
    #     "image",
    #     (1, 3, 376, 1241),
    #     (1, 3, 376, 1241),
    #     (1, 3, 376, 1241),
    # )
    return profile

def build_tensorrt_network(
    args: Namespace,
    trt_logger: trt.Logger,
):
    builder = trt.Builder(trt_logger,)
    config = builder.create_builder_config()
    profile=create_profile_with_shapes(builder)
    config.add_optimization_profile(profile)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(args.model_onnx_path)

    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
    if args.precision_mode == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)

    config.builder_optimization_level=args.optimization_level
    print("Building engine. This might take a while .........")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Engine built, time taken: {duration:.2f}s.")
    with open(args.model_trt_path, "wb") as output_file:
        output_file.write(serialized_engine)
def main():
    args = parse_args()
    trt_logger = LOGGER_DICT["verbose"]
    build_tensorrt_network(args, trt_logger,)



if __name__ == "__main__":
    main()
