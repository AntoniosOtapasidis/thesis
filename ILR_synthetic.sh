#!/bin/bash

RSCRIPT=/users/antonios/miniconda3/envs/r-metabo/bin/Rscript

# Run from the synthetic directory
cd "$(dirname "$0")/R/synthetic_generation"

# Step 1: Generate synthetic community data
echo "=== Step 1: Generating synthetic community ==="
$RSCRIPT complex_synthetic_community.R

# Step 2: CLR transformation
echo "=== Step 2: CLR transformation ==="
$RSCRIPT CLR.R

# Step 3: ILR transformation
echo "=== Step 3: ILR transformation ==="
$RSCRIPT ILR.R

echo "=== Pipeline complete ==="
