#!/bin/bash

# Activate conda environment
source /users/antonios/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# Run the LEAF training script
python run_leaf.py --dataset microbiome_synthetic --num_epoch_s1 30 --num_epoch_s2 30 --iters_pred 10000
