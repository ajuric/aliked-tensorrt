# ALIKED TensorRT

This is an TensorRT implementation of [ALIKED](https://github.com/Shiaoming/ALIKED).

Conversion approach includes two "tricks":
* Adds support for custom DeformConv onnx conversion due to onnx opset version mismatch: torch.onnx currently supports opset 18 (see [here](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html#export-a-pytorch-model-to-onnx)), while DeformConv was added in [opset 19](https://onnx.ai/onnx/operators/onnx__DeformConv.html#l-onnx-doc-deformconv). Custom DeformConv was adopted from [here](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter).
* Instead of using get_patches from custom_ops, get_patches was implemented in PyTorch with available operations.

### Top-K or Score Threshold

Model has two options when filtering keypoints: either you get top-k keypoints, accoding to score, or you get all the keypoints with score larger than threshold.
Since top-k approach gives fixed number of outputs, this approach is selected for TensorRT conversion: thresholding results in unknown number of outputs and is tricky for TensorRT conversion.

Additionally, after fetching top-k keypoints, user can always manually reject those with score lower then threshold.

### Tools

For PyTorch and TensorRT:
* **container**: nvcr.io/nvidia/pytorch:23.12-py3
* **PyTorch**: 2.2.0a0+81ea7a4
* **TensorRT**: 8.6.1
* **Torch-TensorRT**: 2.2.0a0
* **ONNX**: 1.15.0rc2
* **GPUs**: GTX 1660 TI (nvidia-driver: 545.29.06) and RTX 2070 (nvidia-driver:530.41.03)

For PyTorch `torch.compile()` (different container because torch.compile() is in PyTorch nightly build):
* **container**: nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
* **PyTorch**: 2.2.1+cu118
* **GPUs**: GTX 1660 TI (nvidia-driver: 545.29.06)

## Timings

Timings are currently done using data from `assets`. Before measuring inference 
time, models go through warm-up - do couple of inferences in order to fully 
initialize memory on GPU. This initialization takes time, and first inferences take longer than usual because of it. Hence, after doing warm-up, there is no need to reject first few timing measurements.

Rows:
* `model` - ALIKED model type
* `K` - top_k
* `image`: Image sizes of 640x480 and 1241x376 reflect TUM and Kitti image sizes from `assets` dir.

Columns:
* `TRT` - TensorRT
* `PyT` - PyTorch
* `PyT.c` - PyTorch with `torch.compile()`
* `ms` - Mean inference time in miliseconds.
* `MiB` - GPU memory consumption as shown by `nvidia-smi`.

|                                      | GTX 1660 Ti Mobile |  GTX 1660 Ti  |   GTX 1660 Ti   |   RTX 2070   |
|:------------------------------------:|:------------------:|:-------------:|:---------------:|:------------:|
|                                      |    TRT (ms,MiB)    | PyT (ms, MiB) | PyT.c (ms, MiB) | TRT (ms,MiB) |
| model=t16, K=1000, image=640x480     |     11.62, 356     |   15.13, 866  |    10.52, 740   |   9.73, 404  |
| model=t16, K=2000, image=640x480     |     13.55, 364     |   16.32, 858  |    10.75, 792   |  11.04, 404  |
| model=t16, K=1000, image=1241x376    |     15.88, 468     |  19.20, 1222  |   14.23, 1110   |  13.33, 532  |
| model=t16, K=2000, image=1241x376    |     18.76, 474     |  20.30, 1240  |   14.66, 1114   |  15.01, 526  |
| model=n16rot, K=1000, image=640x480  |     17.66, 558     |  20.99, 1490  |   14.81, 1240   |  14.42, 600  |
| model=n16rot, K=2000, image=640x480  |     21.72, 552     |  24.58, 1514  |   15.00, 1314   |  17.18, 604  |
| model=n16rot, K=1000, image=1241x376 |     25.42, 788     |  27.28, 2204  |   21.39, 1884   |  21.86, 818  |
| model=n16rot, K=2000, image=1241x376 |     29.53, 782     |  30.48, 2228  |   21.78, 1932   |  23.53, 824  |


## Convert to ONNX and TensorRT.

To convert model to onnx from pytorch:

```bash
$ python convert_pytorch_to_onnx.py \
    assets/tum \
    --model aliked-n16rot \
    --model_output converted_model/aliked-n16rot-top1k-tum.onnx \
    --opset_version 17 \
    --verbose \
    --top_k 1000
```

To convert model from onnx to TensorRT:

```bash
$ python convert_onnx_to_trt.py \
    --model_onnx_path converted_model/aliked-n16rot-top1k-tum.onnx \
    --model_trt_path converted_model/aliked-n16rot-top1k-tum.trt
```

## Measure

To measure timings, use `measure_timings.py` script which accepts same args as 
`demo_pair.py`

## TODO

* ✅ <s>Refactor code for easier speed measuring</s>.
* ✅ <s>Add warm-up</s>.
* ✅ <s>Add auto-fetching of gpu memory consumption</s>.
* Measure speed and memory in: 
    * ✅ <s>tensorrt</s>
    * ✅ <s>origial pytorch</s>
    * ✅ <s> pytorch.compile</s>
    * pytorch-tensorrt
    * onnx-gpu
* Investigate the speed of custom_ops get_patches and my get_patches.
* Add more data for measuring.
* Add more measurements.
* Add C++ impl.
* Sort outputs before comparing, during conversion in onnx.
* Use NVIDIA Triton inference?
