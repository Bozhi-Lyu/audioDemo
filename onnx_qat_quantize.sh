#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# 1. QAT in Pytorch and then export to ONNX format
# 1.1 QAT the fp32 checkpoint
# python src/onnx_qat_export.py \
#     --config configs/cnn_qat.yaml \
#     --checkpoint models/cnn_fp32_model.pth \
#     --output models/cnn_qat.onnx

# 1.2 Use pretrained QAT checkpoint to export the model to ONNX format
python src/onnx_qat_export.py \
    --config configs/cnn_qat.yaml \
    --checkpoint models/cnn_fp32_model.pth \
    --PretrainedQATCheckpoint models/cnn_qat_for_ONNX.pth \
    --output models/cnn_qat_UsePretrained.onnx

# 2. Evaluate the quantized model before and after exporting
python src/evaluate.py \
    --checkpoint models/cnn_qat_for_ONNX.pth \
    --config configs/cnn_qat.yaml \

python src/onnxRT_inference.py \
    --model models/cnn_qat_UsePretrained.onnx