#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# 1. Export the FP model to ONNX format
# python src/FP_export.py \
#     --config configs/cnn_fp32.yaml \
#     --checkpoint models/cnn_fp32_model.pth \
#     --output models/cnn_fp32.onnx

# 2. Preprocess the ONNX model
python -m onnxruntime.quantization.preprocess \
    --input models/cnn_fp32.onnx \
    --output models/cnn_fp32_infer.onnx

# 3. Perform static quantization
python src/onnx_static_quantize.py \
    --input models/cnn_fp32_infer.onnx \
    --output models/cnn_int8.onnx \
    --per_channel True

# 4. Evaluate the quantized model
# python src/onnxRT_inference.py \
#     --model models/cnn_fp32.onnx
python src/onnxRT_inference.py \
    --model models/cnn_int8.onnx

#################################################
#Without Preprocessing
# 1. Export the FP model to ONNX format

# 2. Perform static quantization
# python src/onnx_static_quantize.py \
#     --input models/cnn_fp32.onnx \
#     --output models/cnn_int8_NoPreProcessing.onnx \
#     --per_channel True\

# # 3. Evaluate the quantized model
# python src/onnxRT_inference.py \
#     --model models/cnn_int8_NoPreProcessing.onnx