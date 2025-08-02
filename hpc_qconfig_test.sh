#!/bin/bash

#BSUB -J qconfig_test
#BSUB -n 4
#BSUB -q hpc
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o qconfig_test_%J.out
#BSUB -e qconfig_test_%J.err

module unload python3
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate audioml

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate audioml

python -m pip install -e . -q
python PTQ_calibration_test.py
