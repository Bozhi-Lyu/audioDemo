#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# 1. PTQ in Pytorch and then export to ONNX format
# 1.1 PTQ the fp32 checkpoint
python src/onnx_ptq_export.py \
    --config configs/cnn_ptq.yaml \
    --checkpoint models/cnn_fp32_model.pth \
    --output models/cnn_ptq.onnx

# 1.2 Use pretrained QAT checkpoint to export the model to ONNX format
# python src/onnx_ptq_export.py \
#     --config configs/cnn_ptq.yaml \
#     --checkpoint models/cnn_fp32_model.pth \
#     --PretrainedQATCheckpoint models/cnn_ptq_for_ONNX.pth \
#     --output models/cnn_ptq_UsePretrained.onnx

# 2. Evaluate the quantized model before and after exporting
python src/evaluate.py \
    --checkpoint models/cnn_ptq_for_ONNX.pth \
    --config configs/cnn_ptq.yaml \

python src/onnxRT_inference.py \
    --model models/cnn_ptq.onnx