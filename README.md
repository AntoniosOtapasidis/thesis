# LEAF for Gut Metabolite Variance Decomposition

Code repository for the MSc thesis: *Decomposing Gut Metabolite Variance into Bacteriome, Virome, and Shared Components using LEAF*.

LEAF (Latent Explainability via Additive Functions) is a two-stage framework that (1) disentangles multimodal microbiome data into modality-specific and shared latent representations using contrastive learning, and (2) decomposes metabolite variance into interpretable additive components via functional ANOVA with neural network decoders.

## Repository Structure

```
.
├── DisentangledSSL/          # Stage 1: Contrastive encoder (InfoNCE + orthogonal loss)
│   ├── algorithms.py         # Training loops for Step 1 (shared Zc) and Step 2 (specific Z1, Z2)
│   ├── models.py             # Encoder architectures
│   ├── losses.py             # InfoNCE, orthogonal, HSIC losses
│   ├── dataset.py            # Multi-omic dataset class
│   └── utils.py              # Utilities
│
├── ND/                       # Stage 2: Neural Decomposition decoder (fANOVA)
│   ├── decoder.py            # Additive decoder with zero-sum constraints
│   ├── decoder_multiple_covariates.py
│   ├── encoder.py            # vMF encoder for latent space
│   ├── CVAE.py               # Variance-explained CVAE
│   ├── CVAE_multiple_covariates.py
│   └── helpers.py            # MDMM augmented Lagrangian
│
├── R/                        # R and Quarto analysis scripts
│   ├── synthetic_generation/  # Synthetic data generation pipeline
│   │   ├── complex_synthetic_community.R   # SCM-based synthetic community (main)
│   │   ├── complex_sparse_community.R      # Sparse variant with zero-inflation
│   │   ├── confounder_included.R           # Confounder scenario (age/BMI)
│   │   ├── CLR.R                           # Centered log-ratio transformation
│   │   └── ILR.R                           # Isometric log-ratio transformation
│   ├── ASCA.qmd              # ANOVA-Simultaneous Component Analysis for confounders
│   ├── spls.R                # Sparse PLS analysis
│   ├── spls_analysis.qmd     # sPLS benchmarking analysis
│   └── multiplicative.replacement.qmd  # Zero-handling for compositional data
│
├── run_leaf_non_linear.py    # Main LEAF training script (synthetic + real data)
├── run_leaf_playground.py    # LEAF playground for experimentation
├── leaf_synthetic.py         # Synthetic experiment runner
│
├── ILR_synthetic.sh          # Pipeline: data generation -> CLR -> ILR
├── run_leaf_microbiome.sh    # Shell script for real microbiome LEAF runs
└── run_bayesian_inference.sh # Shell script for Bayesian inference
```

## Synthetic Data Pipeline

Run the full pipeline from data generation through compositional transformation:

```bash
bash ILR_synthetic.sh
```

This executes three steps:
1. `complex_synthetic_community.R` -- generates synthetic bacteriome, virome, and metabolome data from a structural causal model
2. `CLR.R` -- applies centered log-ratio transformation with prevalence filtering and multiplicative replacement
3. `ILR.R` -- applies isometric log-ratio transformation with balanced sequential binary partition

## Running LEAF

Train LEAF on synthetic data with CLR-transformed inputs:

```bash
python run_leaf_non_linear.py --dataset microbiome_synthetic --beta 0.1 --iters_pred 5000
```
Or with ILR-transformed inputs:

```bash
python run_leaf_non_linear.py --dataset COPSAC_clone --beta 0.1 --iters_pred 5000
```
Or with ILR-transformed inputs:

```bash
python run_leaf_non_linear.py --dataset microbiome_synthetic_ilr --beta 0.1 --iters_pred 5000
```


The confounder variant (`confounder_included.R`) extends this to:

```
Y = W1 * Z1 + W2 * Z2 + Ws * Zs + Wc * C + noise
```

```bash
python leaf_synthetic.py --dataset microbiome_synthetic --beta 0.1 --iters_pred 10000 --iters_res 5000
```
