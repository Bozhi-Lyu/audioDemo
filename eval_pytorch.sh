#!/bin/bash

#BSUB -J eval_pytorch
#BSUB -n 4
#BSUB -q hpc
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o eval_pytorch_%J.out
#BSUB -e eval_pytorch_%J.err
source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e .
python src/evaluate.py --checkpoint models/cnn_fp32_model.pth --config configs/cnn_fp32.yaml

python src/evaluate.py --checkpoint models/cnn_qat_model.pth --config configs/cnn_qat.yaml

python src/evaluate.py --checkpoint models/cnn_ptq_model.pth --config configs/cnn_ptq.yaml