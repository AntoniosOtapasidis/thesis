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
│   ├── kendallW.R            # Kendall's W ranking stability analysis
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
python run_leaf_non_linear.py --dataset microbiome_synthetic --beta 0.1 --seed 42
```

Or with ILR-transformed inputs:

```bash
python run_leaf_non_linear.py --dataset microbiome_synthetic_ilr --beta 0.1 --seed 42
```

## Key Concepts

- **DisentangledSSL**: Two-step contrastive encoder. Step 1 learns shared representations (Zc) via InfoNCE. Step 2 learns modality-specific representations (Z1, Z2) conditioned on Zc with orthogonality constraints.
- **Neural Decomposition**: Additive decoder Y = f1(Z1) + f2(Z2) + fc(Zc) with zero-sum constraints enforced via augmented Lagrangian (MDMM), enabling clean variance decomposition.
- **Permutation testing**: Null distributions built by retraining the decoder on shuffled latent-outcome pairings to assess statistical significance of variance attributions.
- **Kendall's W**: Concordance analysis across random seeds to assess ranking stability per metabolite.

## Generative Model

The synthetic data follow the structural causal model:

```
Y = W1 * Z1 + W2 * Z2 + Ws * Zs + noise
```

Where:
- **Z1, Z2, Zs**: Bacteria-specific, virus-specific, and shared latent factors
- **W1 = -(Cij %*% A1_s)**: Bacteria-to-metabolite weight matrix (via consumption rates)
- **W2 = -(C_virus %*% A2_s)**: Virus-to-metabolite weight matrix (via AMG effects)
- **Ws**: Random Gaussian weight matrix for shared effects
- **A1_s, A2_s**: Loading matrices mapping latent factors to observed taxa (X1, X2)

The confounder variant (`confounder_included.R`) extends this to:

```
Y = W1 * Z1 + W2 * Z2 + Ws * Zs + Wc * C + noise
```

## Experiments

### 1. Synthetic COPSAC (`complex_synthetic_community.R`)

Generates a synthetic community matching the COPSAC2000 cohort dimensions (200 bacteria, 200 viruses, 100 metabolites). Uses `set.seed(69)`. Outputs relative abundance data with `*_COPSAC.csv` suffix.

Output files: `X1_bacteria_synthetic_RA_complex_COPSAC.csv`, `X2_viruses_synthetic_RA_complex_COPSAC.csv`, `Y_metabolites_log_synthetic_complex_RA_COPSAC.csv`, `Z{1,2,s}_latents_RA_complex_COPSAC.csv`, `GT_virome_variance_shares_complex_COPSAC.csv`

### 2. Sparse AMG variant (`complex_sparse_community.R`)

Same generative model but with sparse AMG assignment (1-3 AMGs per virus). Uses `set.seed(69)`. Outputs with `*_complex_sparse.csv` suffix.

Output files: `X1_bacteria_synthetic_RA_complex_sparse.csv`, `X2_viruses_synthetic_RA_complex_sparse.csv`, `Y_metabolites_log_synthetic_complex_sparse.csv`, `Z{1,2,s}_latents_RA_complex_sparse.csv`, `GT_virome_variance_shares_complex_sparse.csv`

### 3. Confounder experiment (`confounder_included.R`)

Extends the sparse model with a confounding variable (C) that has a direct effect on metabolites via Wc. Uses `set.seed(123)`. Also generates confounded latent representations (Z_tilde). Outputs with `*_confounders.csv` suffix.

Output files: `X1_bacteria_synthetic_RA_complex_sparse_confounders.csv`, `X2_viruses_synthetic_RA_complex_sparse_condounders.csv`, `Y_metabolites_log_synthetic_complex_sparse_confounders.csv`, `Z{1,2,s}_latents_RA_complex_sparse_confounders.csv`, `Z{1,2,s}_tilde_latents_RA_complex_sparse_confounders.csv`, `confounders_vector_complex_sparse.csv`, `GT_virome_variance_shares_complex_sparse_confounders.csv`

### 4. Non-linear variant (`non_linear.R`)

Non-linear extension of the confounder model. **Note:** outputs the same filenames as `confounder_included.R` (`*_confounders.csv`), so they will overwrite each other if run in the same directory.

## Datasets

Each script generates:
- **X1**: Bacterial relative abundances (200 bacteria, N=1000 samples)
- **X2**: Viral relative abundances (200 viruses, N=1000 samples)
- **Y**: Log-transformed metabolite concentrations (100 metabolites)
- **Z1, Z2, Zs**: Ground-truth latent factors
- **GT_virome_variance_shares**: Ground-truth variance decomposition per metabolite
