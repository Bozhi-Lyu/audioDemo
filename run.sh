#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e .
# python  ./src/main.py --config ./configs/cnn_fp32.yaml
python  ./src/main.py --config ./configs/cnn_qat.yaml
python ./src/evaluate.py