#!/bin/bash

# Activate conda environment
source /users/antonios/miniconda3/etc/profile.d/conda.sh
conda activate r-metabo

# Run the Bayesian inference R script
Rscript /users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/ILR_SYNTHETIC/ILR.R
