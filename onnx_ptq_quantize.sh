#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# 1. Start from a fp32 checkpoint
# 1.1 Load and PTQ the fp32 checkpoint, then export to ONNX
python src/onnx_ptq_export.py \
    --config configs/cnn_ptq.yaml \
    --checkpoint models/cnn_fp32_model.pth \
    --output models/cnn_ptq.onnx

# 1.2 Evaluate the quantized model before and after exporting
python src/evaluate.py \
    --checkpoint models/cnn_ptq_for_ONNX.pth \
    --config configs/cnn_ptq.yaml \

python src/onnxRT_inference.py \
    --model models/cnn_ptq.onnx

# # 2. Use a PTQ checkpoint pretrained in Pytorch
# # 2.1 Load the PTQ checkpoint and export to ONNX
# python src/onnx_ptq_export.py \
#     --config configs/cnn_ptq.yaml \
#     --PretrainedPTQCheckpoint models/cnn_ptq_for_ONNX.pth \
#     --output models/cnn_ptq_UsePretrained.onnx

# # 2.2 Evaluate the quantized model before and after exporting
# python src/evaluate.py \
#     --checkpoint models/cnn_ptq_for_ONNX.pth \
#     --config configs/cnn_ptq.yaml \

# python src/onnxRT_inference.py \
#     --model models/cnn_ptq_UsePretrained.onnx