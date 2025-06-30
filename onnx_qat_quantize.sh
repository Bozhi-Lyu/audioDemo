#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q

# 1. QAT in Pytorch and then export to ONNX format
python src/onnx_qat_export.py \
    --config configs/cnn_qat.yaml \
    --checkpoint models/cnn_fp32_model.pth \
    --output models/cnn_qat.onnx

# 2. Evaluate the quantized model
python src/onnxRT_inference.py \
    --model models/cnn_qat.onnx