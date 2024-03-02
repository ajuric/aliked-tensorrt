# ALIKED TensorRT

This is an TensorRT implementation of [ALIKED](https://github.com/Shiaoming/ALIKED).

Conversion approach includes two "tricks":
* Adds support for custom DeformConv onnx conversion due to onnx opset version mismatch: torch.onnx currently supports opset 18 (see [here](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html#export-a-pytorch-model-to-onnx)), while DeformConv was added in [opset 19](https://onnx.ai/onnx/operators/onnx__DeformConv.html#l-onnx-doc-deformconv). Custom DeformConv was adopted from [here](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter).
* Instead of using get_patches from custom_ops, get_patches was implemented in PyTorch with available operations.

**NEW!** See `fp16` (AMP) results below!

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

|                                      | GTX 1660 Ti Mobile |  GTX 1660 Ti Mobile |   GTX 1660 Ti  Mobile |   RTX 2070   |
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

## Timings for FP16

Following table shows results for **PyTorch AMP** (Automatix Mixed Precision) 
and **TensorRT** (also utilizing half precision) inference.

By utilising AMP, model uses less memory due to some weights being converted to
half-precision data type (fp16), and, potentially, the inference is faster.


### FP16 slower and GTX-1660-Ti Tensor Cores?
<!-- Add links for these info about Tensor Cores. -->
Note: Measurements are done on GTX-1660-Ti which doesnt't have Tensor Cores 
which are needed for utilizing half-precision data types and making the
inference faster. 

Hence, PyTorch execution of AMP comes with smaller footprint when compared to 
original PyTorch, but the inference is slower.

On the other hand, TensorRT achieves faster inference and smaller memory 
footprint even on GTX 1660 Ti.

### Data Precision Loss?

Even though AMP tries to reduce the precision loss, you still get different 
outputs between original and AMP model.

After visualizing the outputs (keypoints) and comparing the outputs of both 
models (original and AMP), it can be seen that differences are negligible.

The interesting part when comparing the outputs are descriptors, since they
are, actually, multidimensional vectors. When compared in a element-wise 
manner, results are not optimal - there are differences. But when you inspect
the vector similarity by dot product, all their dot products are ~1 
(descriptors here are normalized).

That means that vector directions are preserved and that those descriptors are 
still usefull.

But it would be good to check it on downstream tasks like homography, pose ... 
(in TODOs).

Check the `compare_fp32_fp16.py` for ouputs visualization and comparison.

### Sort the outputs before comparing

It's important to sort the data before comparing, because due to 
half-precision roundings, models output different keypoints: e.g., in 
top_k=1000 config, AMP model misses only ~3 keypoints which the original model
calculated.

Hence, by calculating the mutual nearest neighbours of keypoints (by the x,y coordinates), we find the keypoints common in both outputs.

Check the `find_mutual_closest_keypoints()` method in `compare_fp32_fp16.py`.

### Timigs for FP16

|                                      | GTX 1660 Ti Mobile | GTX 1660 Ti Mobile | GTX 1660 Ti Mobile |
|:------------------------------------:|:------------------:|:------------------:|:------------------:|
|                                      |  TRT.AMP (ms,MiB)  |    PyT (ms, MiB)   |  PyT.AMP (ms, MiB) |
| model=t16, K=1000, image=640x480     |      9.17, 280     |     15.13, 866     |     21.93, 544     |
| model=t16, K=2000, image=640x480     |     10.84, 280     |     16.32, 858     |     25.07, 546     |
| model=t16, K=1000, image=1241x376    |         TBM        |     19.20, 1222    |     28.03, 796     |
| model=t16, K=2000, image=1241x376    |         TBM        |     20.30, 1240    |     30.96, 812     |
| model=n16rot, K=1000, image=640x480  |         TBM        |     20.99, 1490    |     35.85, 1020    |
| model=n16rot, K=2000, image=640x480  |         TBM        |     24.58, 1514    |     44.04, 1052    |
| model=n16rot, K=1000, image=1241x376 |         TBM        |     27.28, 2204    |     47.46, 1536    |
| model=n16rot, K=2000, image=1241x376 |         TBM        |     30.48, 2228    |     54.86, 1542    |


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
    * ✅ <s> pytorch.amp</s>
    * ✅ <s> tensorrt + pytorch.amp</s>
    * pytorch-tensorrt
    * onnx-gpu
* ✅ <s> Sort outputs before comparing, during conversion in onnx. </s>
* Investigate the speed of custom_ops get_patches and my get_patches.
* Add more data for measuring.
* Add more measurements.
* Add C++ impl.
* Use NVIDIA Triton inference?
* Check AMP data on down-stream tasks (e.g. homography, pose, ...)
