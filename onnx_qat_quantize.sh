#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# # 1. Load and QAT the fp32 checkpoint and then export to ONNX format
# python src/onnx_qat_export.py \
#     --config configs/cnn_qat.yaml \
#     --checkpoint models/cnn_fp32_model.pth \
#     --output models/cnn_qat.onnx

# # 2. Evaluate the quantized model before and after exporting
# python src/evaluate.py \
#     --checkpoint models/cnn_qat_for_ONNX.pth \
#     --config configs/cnn_qat.yaml \

# python src/onnxRT_inference.py \
#     --model models/cnn_qat.onnx

# 2. Use a pretrained QAT checkpoint, export then evaluate
python src/onnx_qat_export.py \
    --config configs/cnn_qat.yaml \
    --PretrainedQATCheckpoint models/cnn_qat_for_ONNX.pth \
    --output models/cnn_qat_UsePretrained.onnx

python src/evaluate.py \
    --checkpoint models/cnn_qat_for_ONNX.pth \
    --config configs/cnn_qat.yaml \

python src/onnxRT_inference.py \
    --model models/cnn_qat_UsePretrained.onnx