################################################################################
#                                                                              #
#                   SYNTHETIC MICROBIOME DATA GENERATION                       #
#                   Bacteria-Virus-Metabolite Interactions                     #
#                                                                              #
################################################################################

# ============================================================
# LIBRARIES AND SETUP
# ============================================================
cat("\n=== LOADING LIBRARIES ===\n")
library(deSolve)
#library(rstan)
library(tidyr)
library(ggplot2)
library(dplyr)
library(posterior)
library(stringr)
library(compositions)
cat("All libraries loaded successfully\n")


# SHARED LATENT FACTORS (Zs) 
################################################################################
# 
# Zs represents environmental and host factors that affect BOTH:
# - Bacterial community composition (via As1)
# - Viral community composition (via As2)
# - Metabolite outcomes (via Ws, a random Gaussian weight matrix)
#
################################################################################



################################################################################
# Y GENERATION MODE
################################################################################
# Controls how metabolite outcomes Y are generated from latents Z and observations X
#
# "linear": Y = W1·Z1 + W2·Z2 + Ws·Zc + ε
#           - LEAF-compatible (correct Z→X→Y flow)
#           - Y depends ONLY on latents Z
#           - Mechanistic biology encoded in W matrices
#           - No X1×X2 interactions
#

################################################################################
Y_generation_mode <- "linear"  # Options: "linear", "mechanistic"

cat(sprintf("\n=== Y GENERATION MODE: %s ===\n", Y_generation_mode))
if (Y_generation_mode == "linear") {
  cat("Using LEAF-compatible linear model: Y = W1·Z1 + W2·Z2 + Ws·Zc + ε\n")
  cat("Mechanistic biology (Cij, C_virus, Infection) encoded in W matrices\n")
} else if (Y_generation_mode == "mechanistic") {
  stop("ERROR: Mechanistic mode not yet implemented. Use 'linear' mode.")
}

N_met <- 100   # Number of metabolites
N_bac <- 100   # Number of bacteria
N_vir <- 100   # number of viruses (X2)
 # Reps <- 1     # Number of "experimental" replicates
 # Obs <- 1      # Number of "experimental" observations
 # k <- 14       # Carrying capacity
delta <-1e-6  # Small number to ensure numerical stability
#  C0 <- 0.1     # Metabolism conversion coefficient
n_sims <- 1000  # Number of independent simulations, each simulation uses a separate set of Cij values, this is a different community/sample
set.seed(69)
B_min <- 0.01
B_max <- 1
V_min <- 0.01
V_max <- 1
M_min <- 0.1
target_frac <- 0.1   # max expected depletion is 10 percent of minimal M0
detection_limit <- 1e-4  # Minimum detectable relative abundance for X1 and X2  
cat(sprintf("N_met: %d, N_bac: %d, N_vir: %d\n", N_met, N_bac, N_vir))
cat(sprintf("Number of simulations: %d\n", n_sims))
cat(sprintf("Random seed: 69\n"))

################################################################################
# LATENT FACTOR STRUCTURE DEFINITION
################################################################################
# Define dimensions for the three types of latent factors:
# - Z1: Bacteria-specific factors (captures bacteria-unique variance)
# - Z2: Virus-specific factors (captures virus-unique variance)
# - Zs: Shared factors
################################################################################

#define Z dimensions (latent factors for dimensionality reduction)
cat("\n=== DEFINING LATENT FACTOR DIMENSIONS ===\n")

dim_Z1_s <- 10  # Bacteria-specific latent factors (independent of viruses, conditioned on Zc)
dim_Z2_s <- 10  # Virus-specific latent factors (independent of bacteria, conditioned on Zc)
dim_Zc <- 10  # Shared/common latent factors (affects both bacteria and viruses)

cat(sprintf("dim_Zc (shared/common): %d, dim_Z1_s (bacteria-specific): %d, dim_Z2_s (virus-specific): %d\n",
            dim_Zc, dim_Z1_s, dim_Z2_s))




# ============================================================
# LOADING MATRICES (A matrices map latent factors to taxa)
# ============================================================


# Loading matrices map latent factors to observed taxa (X1, X2)
# Bacteria-specific: Z1_s - X1


 # We are taking the positive loadings from a Guassian distribution with positive values
A1_s <- matrix(rnorm(N_bac * dim_Z1_s, sd = 1), 
               nrow = N_bac, ncol = dim_Z1_s)

# Virus-specific: Z2_s - X2
A2_s <- matrix(rnorm(N_vir * dim_Z2_s, sd = 1), 
               nrow = N_vir, ncol = dim_Z2_s)

# # Shared: Zc - BOTH bacteria and viruses (SAME matrix = disentanglement key)
As1_shared <- matrix(rnorm(N_bac * dim_Zc, sd = 1),
                       nrow = N_bac, ncol = dim_Zc)

As2_shared <- matrix(rnorm(N_vir * dim_Zc, sd = 1),
                     nrow = N_vir, ncol = dim_Zc)


As_shared <- matrix(rnorm(N_bac * dim_Zc, sd=1), nrow=N_bac, ncol=dim_Zc)


cat(sprintf("\n=== Loading Matrices (latents -> taxa) ===\n"))
cat(sprintf("  A1_s: %d x %d (bacteria-specific latents → bacteria)\n",
            nrow(A1_s), ncol(A1_s)))
cat(sprintf("  A2_s: %d x %d (virus-specific latents → viruses)\n",
            nrow(A2_s), ncol(A2_s)))
# cat(sprintf("  Ac: %d x %d (shared → metabolites)\n",
#             nrow(Ac), ncol(Ac)))


cat("\n=== COMPUTING WEIGHT MATRICES (W1, W2, Ws) FROM CONSUMPTION ===\n")


################################################################################
# VIRUS-BACTERIA INFECTION MATRIX PROOF OF CONCEPT 1+1 BACTERIA VIRUS PAIRS
################################################################################
# Each viral species has 1 bacterial host of bacteria with varying strength
################################################################################
# cat("\n=== CREATING VIRUS-BACTERIA INFECTION MATRIX ===\n")

# if (N_vir != N_bac) {
#   stop("For 1–1 virus–bacterium pairs, require N_vir == N_bac")
# }

# # Each virus infects exactly one unique bacterium, and vice versa
# Infection <- diag(1, nrow = N_vir, ncol = N_bac)

# non_zero_infections <- sum(Infection > 0)
# cat(sprintf("Infection matrix dimensions: %d x %d\n", nrow(Infection), ncol(Infection)))
# cat(sprintf("Total infections: %d (%.2f%% of matrix)\n",
#             non_zero_infections, 100 * non_zero_infections / (N_vir * N_bac)))
# cat(sprintf("Average bacteria infected per virus: %.1f\n", mean(rowSums(Infection > 0))))
# cat(sprintf("Average viruses per bacterium: %.1f\n", mean(colSums(Infection > 0))))


################################################################################
# VIRUS-BACTERIA INFECTION MATRIX PROOF OF CONCEPT 1 VIRUS 1 BACTERIA BUT 1 BACTERIA MULTIPLE PARACITES
################################################################################

# cat("\n=== CREATING VIRUS-BACTERIA INFECTION MATRIX COPSAC CLONE===\n")

# # Infection[i_virus, j_bacterium] = 1 if virus i infects bacterium j
# Infection <- matrix(0, nrow = N_vir, ncol = N_bac)

# # Each virus picks exactly one host bacterium (with replacement)
# host_ids <- sample(1:N_bac, size = N_vir, replace = TRUE)

# for (v in 1:N_vir) {
#   Infection[v, host_ids[v]] <- 1
# }

# non_zero_infections <- sum(Infection > 0)
# cat(sprintf("Infection matrix dimensions: %d x %d\n", nrow(Infection), ncol(Infection)))
# cat(sprintf("Total infections: %d (%.2f%% of matrix)\n",
#             non_zero_infections, 100 * non_zero_infections / (N_vir * N_bac)))
# cat(sprintf("Average bacteria infected per virus: %.2f\n", mean(rowSums(Infection > 0))))
# cat(sprintf("Average viruses per bacterium: %.2f\n", mean(colSums(Infection > 0))))


################################################################################
# METABOLITE GROUP DEFINITIONS
################################################################################
# Partition metabolites into 4 groups based on what drives their variance:
# - met_B (1-30):    Bacteria-dominated metabolites
# - met_V (31-60):   Virus-dominated metabolites (via AMGs)
# - met_S (61-90):   Shared metabolites (both bacteria and virus effects)
# - met_N (91-100):  Noise metabolites (minimal signal)
################################################################################

# ============================================================
# METABOLITE GROUP DEFINITIONS
# ============================================================
cat("\n=== DEFINING METABOLITE GROUPS ===\n")

# 0) Which metabolites are consumable
consumed_metabolites <- 1:N_met

## ---------- metabolite groups by index ----------
# Four main groups based on what drives their variance:
met_B <- 1:30            # bacteria-dominated
met_V <- 31:60           # virus-dominated
met_S <- 61:90           # shared (both bacteria and virus)
met_N <- 91:100          # noise / null (minimal signal)

is_B <- 1:N_met %in% met_B
is_V <- 1:N_met %in% met_V
is_S <- 1:N_met %in% met_S
is_N <- 1:N_met %in% met_N

cat(sprintf("Metabolite groups defined:\n"))
cat(sprintf("  - Bacteria-dominated (met_B): %d metabolites (%d-%d)\n",
            length(met_B), min(met_B), max(met_B)))
cat(sprintf("  - Virus-dominated (met_V): %d metabolites (%d-%d)\n",
            length(met_V), min(met_V), max(met_V)))
cat(sprintf("  - Shared (met_S): %d metabolites (%d-%d)\n",
            length(met_S), min(met_S), max(met_S)))
cat(sprintf("  - Noise/Null (met_N): %d metabolites (%d-%d)\n",
            length(met_N), min(met_N), max(met_N)))



################################################################################
# BACTERIA-METABOLITE CONSUMPTION MATRIX (Cij)
################################################################################
# Defines baseline bacterial consumption rates for each metabolite
# Cij[i, j] = rate at which bacterium j consumes metabolite i
# Different metabolite groups have different baseline consumption rates:
# - met_B: High baseline (bacteria-driven)
# - met_V: Low baseline (will be boosted by viral AMGs)
# - met_S: Moderate baseline (both contribute)
# - met_N: Zero (no bacterial interactions)
################################################################################

# ============================================================
# BACTERIA-METABOLITE INTERACTION MATRIX (Cij) - DENSE WITH METABOLIC CLUSTERS
# ============================================================
cat("\n=== BUILDING BACTERIA-METABOLITE INTERACTION MATRIX (Cij) ===\n")

# Cij[i,j] represents how much bacterium j consumes metabolite i
Cij <- matrix(0, N_met, N_bac)

# Consumption weight base
consumption_weight  <- 0.1
cat(sprintf("Base consumption rate (consumption_weight): %.6f\n", consumption_weight))

# Assign bacteria to 4 metabolic profile clusters
n_metabolic_clusters <- 4
bac_cluster <- sample(1:n_metabolic_clusters, N_bac, replace = TRUE)
cat(sprintf("Assigned %d bacteria to %d metabolic clusters\n", N_bac, n_metabolic_clusters))
cat(sprintf("Cluster distribution: %s\n", paste(table(bac_cluster), collapse = ", ")))

# Define cluster preferences for metabolite groups
# Each cluster has different affinities for met_B, met_V, met_S
cluster_prefs <- list(
  cluster1 = list(B = 0.8, V = 0.4, S = 0.6),  # Prefers bacteria-dominated
  cluster2 = list(B = 0.5, V = 0.7, S = 0.6),  # Prefers virus-related
  cluster3 = list(B = 0.6, V = 0.6, S = 0.8),  # Prefers shared
  cluster4 = list(B = 0.6, V = 0.5, S = 0.5)   # Balanced/generalist
)

# met_N (91-100) = no bacterial interactions
Cij[met_N, ] <- 0
cat(sprintf("Set %d noise metabolites to ZERO bacterial interactions\n", length(met_N)))

# Build dense interactions for 90 signal metabolites
signal_metabolites <- c(met_B, met_V, met_S)  # 90 metabolites
cat(sprintf("Building dense interactions for %d signal metabolites\n", length(signal_metabolites)))

for (b in 1:N_bac) {
  # Each bacterium interacts with 50-80 metabolites (out of 90)
  n_interactions <- sample(50:80, 1)

  # Get cluster preferences
  cluster_id <- bac_cluster[b]
  prefs <- cluster_prefs[[cluster_id]]

  # Create probability weights for sampling metabolites based on cluster
  prob_weights <- numeric(90)
  prob_weights[1:30]   <- prefs$B  # met_B preference
  prob_weights[31:60]  <- prefs$V  # met_V preference
  prob_weights[61:90]  <- prefs$S  # met_S preference

  # Add moderate random variation (not too deterministic)
  prob_weights <- prob_weights + runif(30, -0.2, 0.2)
  prob_weights <- pmax(prob_weights, 0.1)  # Ensure all have some chance

  # Sample which metabolites this bacterium interacts with
  interacting_mets <- sample(signal_metabolites, size = n_interactions,
                              replace = FALSE, prob = prob_weights)

  # Assign weights based on metabolite group and cluster
  for (met in interacting_mets) {
    if (met %in% met_B) {
      # Bacteria-dominated: higher baseline
      weight <- runif(1, min = 0.05 * consumption_weight,
                         max = 1.0 * consumption_weight)
    } else if (met %in% met_V) {
      # Virus-dominated: weaker baseline (will be boosted by AMGs)
      weight <- runif(1, min = 0.02 * consumption_weight,
                         max = 0.6 * consumption_weight)
    } else {  # met_S
      # Shared: moderate baseline
      weight <- runif(1, min = 0.04 * consumption_weight,
                         max = 0.8 * consumption_weight)
    }

    Cij[met, b] <- weight
  }
}

# ===================================================================
# DIAGNOSTICS
# ===================================================================
total_interactions <- sum(Cij > 0)
density <- total_interactions / (N_met * N_bac)
interactions_per_bac <- colSums(Cij > 0)
interactions_per_met <- rowSums(Cij > 0)

cat(sprintf("\nCij Matrix Statistics:\n"))
cat(sprintf("  Total non-zero interactions: %d\n", total_interactions))
cat(sprintf("  Overall density: %.2f%%\n", density * 100))
cat(sprintf("  Interactions per bacterium: mean=%.1f, range=[%d, %d]\n",
            mean(interactions_per_bac), min(interactions_per_bac), max(interactions_per_bac)))
cat(sprintf("  Interactions per metabolite: mean=%.1f, range=[%d, %d]\n",
            mean(interactions_per_met), min(interactions_per_met), max(interactions_per_met)))

# Check by metabolite group
cat(sprintf("\nInteractions by metabolite group:\n"))
cat(sprintf("  met_B (bacteria-dominated): mean=%.1f bacteria per metabolite\n",
            mean(interactions_per_met[met_B])))
cat(sprintf("  met_V (virus-dominated): mean=%.1f bacteria per metabolite\n",
            mean(interactions_per_met[met_V])))
cat(sprintf("  met_S (shared): mean=%.1f bacteria per metabolite\n",
            mean(interactions_per_met[met_S])))
cat(sprintf("  met_N (noise): mean=%.1f bacteria per metabolite (should be 0)\n",
            mean(interactions_per_met[met_N])))
################################################################################
# AUXILIARY METABOLITE GENES (AMG) CONFIGURATION
################################################################################
# AMGs are viral genes that boost metabolite consumption
# This section defines:
# - Which metabolites are eligible for AMG effects (V_amg and S_amg)
# - The strength of AMG effects (min_boost and max_boost)
# Note: vir_V and vir_S are defined but unused; actual AMG assignment
# loops over all viruses and samples uniformly from amg_metabolites
################################################################################

# ============================================================
# AUXILIARY METABOLITE-GENOME (AMG) SETUP
# ============================================================
cat("\n=== SETTING UP AMG (AUXILIARY METABOLITE GENES) ===\n")
# AMGs are viral genes that boost metabolite consumption in infected bacteria

V_amg <- met_V   # Metabolites affected by virus-specific AMGs (31–60)
S_amg <- met_S   # Metabolites affected by shared AMGs (61–90)

vir_V <- 31:60   # Viruses targeting virus-dominated metabolites
vir_S <- 61:90   # Viruses targeting shared metabolites

cat(sprintf("V_amg metabolites: %d-%d (%d total)\n", min(V_amg), max(V_amg), length(V_amg)))
cat(sprintf("S_amg metabolites: %d-%d (%d total)\n", min(S_amg), max(S_amg), length(S_amg)))
cat(sprintf("vir_V: %d viruses (%d-%d)\n", length(vir_V), min(vir_V), max(vir_V)))
cat(sprintf("vir_S: %d viruses (%d-%d)\n", length(vir_S), min(vir_S), max(vir_S)))

# AMG boost strength (how much viruses increase metabolite consumption)
min_boost <- 0.2 * consumption_weight
max_boost <- 0.8 * consumption_weight
cat(sprintf("AMG boost range: [%.6f, %.6f]\n", min_boost, max_boost))



################################################################################
# VIRUS-METABOLITE INTERACTION MATRIX (C_virus / AMG EFFECTS)
################################################################################
################################################################################

###################################################
# VIRUS-METABOLITE INTERACTION MATRIX (C_virus / AMG effects) - SPARSE
###################################################
cat("\n=== BUILDING SPARSE VIRUS-METABOLITE (AMG) INTERACTION MATRIX (C_virus) ===\n")

# Initialize C_virus matrix
# C_virus[i,v] represents AMG boost from virus v on metabolite i
C_virus <- matrix(0, nrow = N_met, ncol = N_vir)
cat(sprintf("Initialized C_virus matrix: %d x %d\n", nrow(C_virus), ncol(C_virus)))

# Define AMG effect types (shared vs overtake)
AMG_effect_type <- matrix("none", nrow = N_met, ncol = N_vir)

# No AMGs for met_B and met_N
C_virus[met_B, ] <- 0
C_virus[met_N, ] <- 0
cat(sprintf("Set C_virus to 0 for %d bacteria-dominated metabolites (met_B)\n", length(met_B)))
cat(sprintf("Set C_virus to 0 for %d noise metabolites (met_N)\n", length(met_N)))

# Build sparse AMG interactions
amg_metabolites <- c(met_V, met_S)  # Only these 60 metabolites can have AMGs
cat(sprintf("\nBuilding sparse AMG interactions from %d eligible metabolites\n", length(amg_metabolites)))

n_shared_amgs <- 0
n_overtake_amgs <- 0

for (v in 1:N_vir) {
  # Each virus has 1-3 AMGs (very sparse!)
  n_amgs <- sample(1:3, 1)

  # Sample which metabolites this virus has AMGs for (uniform random)
  amg_targets <- sample(amg_metabolites, size = n_amgs, replace = FALSE)

  # Assign AMG weights and effect types
  for (met in amg_targets) {
    # Randomly decide: shared effect (60%) or overtake effect (40%)
    effect_type <- sample(c("shared", "overtake"), 1, prob = c(0.6, 0.4))

    if (effect_type == "shared") {
      # Shared effect: moderate boost
      amg_weight <- runif(1, min = 0.04 * consumption_weight,
                              max = 0.8 * consumption_weight)
      n_shared_amgs <- n_shared_amgs + 1
    } else {
      # Overtake effect: strong boost
      amg_weight <- runif(1, min = 0.8 * consumption_weight,
                              max = 2.0 * consumption_weight)
      n_overtake_amgs <- n_overtake_amgs + 1
    }

    C_virus[met, v] <- amg_weight
    AMG_effect_type[met, v] <- effect_type
  }
}

# ===================================================================
# DIAGNOSTICS
# ===================================================================
total_amgs <- sum(C_virus > 0)
amg_density <- total_amgs / (N_met * N_vir)
amgs_per_virus <- colSums(C_virus > 0)
amgs_per_metabolite <- rowSums(C_virus > 0)

cat(sprintf("\nC_virus Matrix Statistics:\n"))
cat(sprintf("  Total AMGs: %d\n", total_amgs))
cat(sprintf("  Overall density: %.2f%% (very sparse)\n", amg_density * 100))
cat(sprintf("  AMGs per virus: mean=%.1f, range=[%d, %d]\n",
            mean(amgs_per_virus), min(amgs_per_virus), max(amgs_per_virus)))
cat(sprintf("  Shared effects (additive): %d (%.1f%%)\n",
            n_shared_amgs, 100 * n_shared_amgs / total_amgs))
cat(sprintf("  Overtake effects (dominant): %d (%.1f%%)\n",
            n_overtake_amgs, 100 * n_overtake_amgs / total_amgs))



#----------------------------------------------------------------------------------------------

################################################################################
# VIRUS-BACTERIA INFECTION MATRIX (SPARSE, BINARY)
################################################################################
# Infection[v, b] = 1 if virus v can infect bacterium b, 0 otherwise
# Key features:
# - Selectivity-based: viruses are categorized into 3 selectivity classes
# - Specialists: infect 1 host (highly selective)
# - Moderate: infect 2-10 hosts (moderately selective)
# - Generalists: infect 10-50 hosts (low selectivity)
# - Multiple viruses can infect the same bacterium (realistic co-infection)
# - Uniform random host selection (no cluster preferences)
################################################################################
cat("\n=== BUILDING SPARSE VIRUS-BACTERIA INFECTION MATRIX ===\n")

# Initialize infection matrix (N_vir x N_bac)
Infection <- matrix(0L, nrow = N_vir, ncol = N_bac)
cat(sprintf("Initialized Infection matrix: %d x %d\n", nrow(Infection), ncol(Infection)))

# Define 3 selectivity classes
n_selectivity_classes <- 3L
selectivity_classes <- c("specialist", "moderate", "generalist")

# Assign each virus to a selectivity class
# Distribution: ~30% specialists, ~50% moderate, ~20% generalists
selectivity_probs <- c(0.3, 0.5, 0.2)
virus_selectivity <- sample(1:n_selectivity_classes, N_vir, replace = TRUE,
                           prob = selectivity_probs)

# Define number of hosts per selectivity class
selectivity_host_ranges <- list(
  specialist = c(1, 1),      # Always 1 host
  moderate = c(2, 10),        # 2-10 hosts
  generalist = c(10,50)       # 10-50 hosts
)

# Build infection matrix
for (v in 1:N_vir) {
  # Determine number of hosts based on selectivity class
  v_class <- virus_selectivity[v]
  host_range <- selectivity_host_ranges[[v_class]]

  n_hosts <- sample(host_range[1]:host_range[2], 1)

  # Uniform random sampling of bacterial hosts (no cluster preference)
  infected_bacteria <- sample(1:N_bac, size = n_hosts, replace = FALSE)

  # Set infection matrix (binary: 1 = can infect, 0 = cannot)
  Infection[v, infected_bacteria] <- 1L
}

# ===================================================================
# DIAGNOSTICS
# ===================================================================
total_infections <- sum(Infection)
infection_density <- total_infections / (N_vir * N_bac)
hosts_per_virus <- rowSums(Infection)
viruses_per_bacterium <- colSums(Infection)

cat(sprintf("\n=== Infection Matrix Statistics ===\n"))
cat(sprintf("  Total infections: %d\n", total_infections))
cat(sprintf("  Overall density: %.2f%% (very sparse!)\n", infection_density * 100))
cat(sprintf("  Hosts per virus: mean=%.1f, range=[%d, %d]\n",
            mean(hosts_per_virus), min(hosts_per_virus), max(hosts_per_virus)))
cat(sprintf("  Viruses per bacterium: mean=%.1f, range=[%d, %d]\n",
            mean(viruses_per_bacterium), min(viruses_per_bacterium), max(viruses_per_bacterium)))

# By construction every virus has at least one host, but keep a safety check
if (any(hosts_per_virus == 0)) {
  warning("WARNING: Some viruses have NO bacterial hosts!")
}

# Check selectivity class distribution
cat(sprintf("\n=== Selectivity Class Distribution ===\n"))
for (sc in 1:n_selectivity_classes) {
  n_viruses_in_class <- sum(virus_selectivity == sc)
  class_name <- selectivity_classes[sc]

  # Calculate average hosts for viruses in this class
  viruses_in_class <- which(virus_selectivity == sc)
  if (length(viruses_in_class) > 0) {
    avg_hosts <- mean(hosts_per_virus[viruses_in_class])
    cat(sprintf("  %s (class %d): %d viruses (%.1f%%), avg hosts = %.1f\n",
                class_name, sc, n_viruses_in_class,
                100 * n_viruses_in_class / N_vir, avg_hosts))
  }
}

cat("\nInfection matrix construction complete!\n")


################################################################################
# INFECTION PROFILE - BIOLOGICAL REALISM
################################################################################
cat("\n=== INFECTION PROFILE SUMMARY ===\n")

# Biological rationale:
# - The Infection matrix defines which bacteria each virus can infect
# - C_virus already contains the AMG effects (viral genes that boost metabolism)
# - AMGs hijack bacterial metabolism when viruses infect their hosts
# - The infection matrix provides biological realism but doesn't modify C_virus
# - C_virus effects are applied through the model: Y = W1*Z1 + W2*Z2 + ...
#   where W2 = C_virus %*% A2_s

# Diagnostics
hosts_per_virus <- rowSums(Infection)
cat(sprintf("  Total viral-bacterial infections: %d\n", sum(Infection)))
cat(sprintf("  Infection density: %.2f%%\n", 100 * sum(Infection) / (N_vir * N_bac)))
cat(sprintf("  Mean hosts per virus: %.2f ± %.2f\n", mean(hosts_per_virus), sd(hosts_per_virus)))
cat(sprintf("  Host range: %d to %d bacteria\n", min(hosts_per_virus), max(hosts_per_virus)))
cat(sprintf("  Total AMGs (virus-metabolite pairs): %d\n", sum(C_virus > 0)))
cat(sprintf("  Mean AMG strength: %.6f\n", mean(C_virus[C_virus > 0])))



# ============================================================
# STORAGE MATRICES FOR SIMULATIONS
# ============================================================
cat("\n=== PREALLOCATING STORAGE MATRICES ===\n")

# Store final abundances/concentrations for each simulation
Y_mat  <- matrix(0, nrow = n_sims, ncol = N_met)  # Metabolite log-abundances
X1_mat <- matrix(0, nrow = n_sims, ncol = N_bac)  # Bacteria relative abundances
X2_mat <- matrix(0, nrow = n_sims, ncol = N_vir)  # Virus relative abundances

# Store latent factors for each simulation
Z1_store <- matrix(0, nrow = n_sims, ncol = dim_Z1_s)  # Bacteria-specific latents
Z2_store <- matrix(0, nrow = n_sims, ncol = dim_Z2_s)  # Virus-specific latents
Zs_store <- matrix(0, nrow = n_sims, ncol = dim_Zc)  # Shared latents

# Storage for sparsity diagnostics
sparsity_B <- numeric(n_sims)  # Percentage of zeros in bacteria
sparsity_V <- numeric(n_sims)  # Percentage of zeros in viruses

cat(sprintf("Allocated storage for %d simulations\n", n_sims))
cat(sprintf("  - Y_mat: %d x %d (metabolites)\n", nrow(Y_mat), ncol(Y_mat)))
cat(sprintf("  - X1_mat: %d x %d (bacteria)\n", nrow(X1_mat), ncol(X1_mat)))
cat(sprintf("  - X2_mat: %d x %d (viruses)\n", nrow(X2_mat), ncol(X2_mat)))
cat(sprintf("  - Sparsity tracking enabled for B_comp and V_comp\n"))


################################################################################
# WEIGHT MATRICES CONSTRUCTION
# W1 = -(Cij %*% A1_s), W2 = -(C_virus %*% A2_s), Ws = random Gaussian
# Y = W1*Z1 + W2*Z2 + Ws*Zs + noise
################################################################################

cat("\n=== COMPUTING WEIGHT MATRICES (Ws, W1, W2) ===\n")



W1 <- -(Cij %*% A1_s)
cat(sprintf("W1 = -Cij * A1: %d x %d (NEGATIVE = consumption)\n", nrow(W1), ncol(W1)))

# # Virus-specific pathway: Z2_s affects metabolites through viral AMGs
# cat(sprintf("\nComputing W2 = -(C_virus (%d x %d)) %%*%% A2_s (%d x %d)\n",
#             nrow(C_virus), ncol(C_virus),
#             nrow(A2_s), ncol(A2_s)))

# #W2 <- - C_virus %*% A2_s
# W2 <- (C_virus %*% A2_s)


# cat(sprintf("W2 = (C_virus) * A2_s: %d x %d (NEGATIVE = consumption)\n", nrow(W2), ncol(W2)))

# Virus-specific pathway: Z2_s affects metabolites through viral AMGs
cat(sprintf("\nComputing W2 from viral AMG effects:\n"))
cat(sprintf("  C_virus: %d x %d (metabolite x virus)\n", nrow(C_virus), ncol(C_virus)))
cat(sprintf("  Infection: %d x %d (virus x bacterium)\n", nrow(Infection), ncol(Infection)))
cat(sprintf("  A2_s: %d x %d (virus x Z2)\n", nrow(A2_s), ncol(A2_s)))

# C_virus encodes AMG effects; Infection matrix kept as metadata
W2 <- -(C_virus %*% A2_s)

cat(sprintf("W2 (virus→metabolites via AMGs): %d x %d\n",
            nrow(W2), ncol(W2)))
cat("Note: Infection matrix constrains which bacteria viruses infect (affects X2),\n")
cat("      but viral AMGs create NEW metabolic capabilities in infected hosts.\n")

# Shared pathway: both bacteria and viruses contribute through shared latent
#Ws <- - (Cij %*% As_shared + C_virus %*% As_shared) # in the case dimensions X1 = X2 
#Ws <- - (Cij %*% As1_shared + C_virus %*% As2_shared) # this is for the COPSAC clone

Ws <- matrix(
  rnorm(N_met * dim_Zc, sd = 1),nrow = N_met,ncol = dim_Zc
)

cat(sprintf("Ws = random Gaussian matrix: %d x %d\n", nrow(Ws), ncol(Ws)))

cat(sprintf("\n=== Weight Matrix Summary ===\n"))
cat(sprintf("W1 (bacteria→metabolites): %d x %d\n", nrow(W1), ncol(W1)))
cat(sprintf("W2 (virus→metabolites): %d x %d\n", nrow(W2), ncol(W2)))
cat(sprintf("Ws (shared→metabolites): %d x %d\n", nrow(Ws), ncol(Ws)))


# Right after Wc, W1_s, W2_s computation, add:
cat("\n=== PRE-SCALING WEIGHT MATRIX CHECK ===\n")
cat(sprintf("Ws row norms: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(sqrt(rowSums(Ws^2))),
            min(sqrt(rowSums(Ws^2))),
            max(sqrt(rowSums(Ws^2)))))
cat(sprintf("W1 row norms: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(sqrt(rowSums(W1^2))),
            min(sqrt(rowSums(W1^2))),
            max(sqrt(rowSums(W1^2)))))
cat(sprintf("W2 row norms: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(sqrt(rowSums(W2^2))),
            min(sqrt(rowSums(W2^2))),
            max(sqrt(rowSums(W2^2)))))

cat("Mechanistic structure fully defined\n")


# ============================================================
# VARIANCE CALCULATION PER METABOLITE GROUP
# ============================================================
# cat("\n=== COMPUTING VARIANCE CONTRIBUTIONS PER GROUP ===\n")
# Calculate how much variance each pathway (W1, W2, Ws) contributes
# to each metabolite group (B, V, S, N)

# Helper function: compute row-wise squared norms
row_sq <- function(M) rowSums(M^2)

# Bacteria-dominated metabolites (met_B)
vB_bac <- mean(row_sq(W1)[met_B])  # Variance from bacteria pathway
vB_vir <- mean(row_sq(W2)[met_B])  # Variance from virus pathway
vB_sh  <- mean(row_sq(Ws)[met_B])  # Variance from shared pathway

# Virus-dominated metabolites (met_V)
vV_bac <- mean(row_sq(W1)[met_V])
vV_vir <- mean(row_sq(W2)[met_V])
vV_sh  <- mean(row_sq(Ws)[met_V])

# Shared metabolites (met_S)
vS_bac <- mean(row_sq(W1)[met_S])
vS_vir <- mean(row_sq(W2)[met_S])
vS_sh  <- mean(row_sq(Ws)[met_S])

# Noise metabolites (met_N)
vN_bac <- if (length(met_N) > 0) mean(row_sq(W1)[met_N]) else 0
vN_vir <- if (length(met_N) > 0) mean(row_sq(W2)[met_N]) else 0
vN_sh  <- if (length(met_N) > 0) mean(row_sq(Ws)[met_N]) else 0

cat("Variance contributions calculated:\n")
cat(sprintf("  Bacteria-dominated (B): bac=%.4f, vir=%.4f, shared=%.4f\n",
            vB_bac, vB_vir, vB_sh))
cat(sprintf("  Virus-dominated (V): bac=%.4f, vir=%.4f, shared=%.4f\n",
            vV_bac, vV_vir, vV_sh))
cat(sprintf("  Shared (S): bac=%.4f, vir=%.4f, shared=%.4f\n",
            vS_bac, vS_vir, vS_sh))
cat(sprintf("  Noise (N): bac=%.4f, vir=%.4f, shared=%.4f\n",
            vN_bac, vN_vir, vN_sh))

# Avoid division by zero
eps  <- 1e-12
safe <- function(x) ifelse(x < eps, eps, x)

# Total variance per group (sum across all pathways)
vB_tot <- safe(vB_bac + vB_vir + vB_sh)
vV_tot <- safe(vV_bac + vV_vir + vV_sh)
vN_tot <- safe(vN_bac + vN_vir + vN_sh)
vS_tot <- safe(vS_bac + vS_vir + vS_sh)

cat(sprintf("\nTotal variance per group:\n"))
cat(sprintf("  - vB_tot: %.4f\n", vB_tot))
cat(sprintf("  - vV_tot: %.4f\n", vV_tot))
cat(sprintf("  - vS_tot: %.4f\n", vS_tot))
cat(sprintf("  - vN_tot: %.4f\n", vN_tot))




################################################################################
# TARGET VARIANCE FRACTIONS (GROUND TRUTH DESIGN)
################################################################################
# Define the desired variance decomposition for each metabolite group
#
# For each group, specify what fraction of variance should come from:
# - bac: bacteria pathway (W1 * Z1)
# - vir: virus pathway (W2 * Z2)
# - sh: shared pathway (Ws * Zs)

################################################################################



cat("\n=== SETTING TARGET VARIANCE FRACTIONS ===\n")

# Define the weight matrices


eps  <- 1e-12
safe <- function(x) ifelse(x < eps, eps, x)

cat("\n=== SETTING TARGET VARIANCE FRACTIONS ===\n")

target_B <- c(bac = 0.8, vir = 0.00, sh = 0.2)  # Stronger shared signal
target_V <- c(bac = 0.1, vir = 0.8, sh = 0.1)  # Stronger shared signal
target_S <- c(bac = 0.2, vir = 0.2, sh = 0.60)  # Balanced
target_N <- c(bac = 0.00, vir = 0.00, sh = 0.00)

cat("Target variance fractions:\n")
cat(sprintf("  Bacteria-dominated (B): bac=%.2f, vir=%.2f, shared=%.2f\n",
            target_B["bac"], target_B["vir"], target_B["sh"]))
cat(sprintf("  Virus-dominated (V): bac=%.2f, vir=%.2f, shared=%.2f\n",
            target_V["bac"], target_V["vir"], target_V["sh"]))
cat(sprintf("  Shared (S): bac=%.2f, vir=%.2f, shared=%.2f\n",
            target_S["bac"], target_S["vir"], target_S["sh"]))
cat(sprintf("  Noise (N): bac=%.2f, vir=%.2f, shared=%.2f\n",
            target_N["bac"], target_N["vir"], target_N["sh"]))

# while keeping metabolite-to-metabolite differences inside the group.
scale_group <- function(indices, targets) {
  if (length(indices) == 0) return()

  eps <- 1e-12

  # Current variances for each metabolite in the group
  v1_s <- rowSums(W1[indices, , drop = FALSE]^2)  # Bacteria-specific variance
  v2_s <- rowSums(W2[indices, , drop = FALSE]^2)  # Virus-specific variance
  vc <- rowSums(Ws[indices, , drop = FALSE]^2)      # Shared/common variance

  # Group-average variances for each pathway
  m1_s <- max(mean(v1_s), eps)
  m2_s <- max(mean(v2_s), eps)
  mc <- max(mean(vc), eps)

  # Desired group-average variances (total signal set to 1 here)
  t1 <- targets["bac"]  # Bacteria-specific target
  t2 <- targets["vir"]  # Virus-specific target
  tc <- targets["sh"]   # Shared/common target

  # Scaling factors for the whole group and pathway
  a1_s <- if (t1 > 0 && m1_s > eps) sqrt(t1 / m1_s) else 0
  a2_s <- if (t2 > 0 && m2_s > eps) sqrt(t2 / m2_s) else 0
  ac <- if (tc > 0 && mc > eps) sqrt(tc / mc) else 0

  # Apply scaling in place. This preserves differences between metabolites
  # inside the group but shifts the whole group toward the target fractions.
  W1[indices, ] <<- W1[indices, ] * a1_s
  W2[indices, ] <<- W2[indices, ] * a2_s
  Ws[indices, ] <<- Ws[indices, ] * ac
}

cat("Applying group-wise scaling...\n")

# B bacteria-dominated metabolites
scale_group(met_B, target_B)

# V virus-dominated metabolites
scale_group(met_V, target_V)

# S shared metabolites
scale_group(met_S, target_S)

# N "noise" metabolites (balanced signal)
scale_group(met_N, target_N)

# Verify achieved fractions at group level
cat("\nVerifying achieved target fractions (group means):\n")

verify_group <- function(indices, group_name, targets) {
  if (length(indices) == 0) return()
  
  v1 <- rowSums(W1[indices, , drop = FALSE]^2)
  v2 <- rowSums(W2[indices, , drop = FALSE]^2)
  vs <- rowSums(Ws[indices, , drop = FALSE]^2)
  total <- v1 + v2 + vs
  
  frac_1 <- mean(v1 / total, na.rm = TRUE)
  frac_2 <- mean(v2 / total, na.rm = TRUE)
  frac_s <- mean(vs / total, na.rm = TRUE)
  
  cat(sprintf("  %s: bac=%.3f (target=%.2f), vir=%.3f (target=%.2f), sh=%.3f (target=%.2f)\n",
              group_name, frac_1, targets["bac"], frac_2, targets["vir"],
              frac_s, targets["sh"]))
}

verify_group(met_B, "met_B", target_B)
verify_group(met_V, "met_V", target_V)
verify_group(met_S, "met_S", target_S)
verify_group(met_N, "met_N", target_N)

# DIAGNOSTIC: Check W2 variance for each group
cat("\n=== W2 DIAGNOSTIC (after group-wise scaling) ===\n")
cat(sprintf("met_B (should be ~0): mean=%.6f, range=[%.6f, %.6f]\n",
            mean(rowSums(W2[met_B, ]^2)), min(rowSums(W2[met_B, ]^2)), max(rowSums(W2[met_B, ]^2))))
cat(sprintf("met_V (should be ~0.8): mean=%.6f, range=[%.6f, %.6f]\n",
            mean(rowSums(W2[met_V, ]^2)), min(rowSums(W2[met_V, ]^2)), max(rowSums(W2[met_V, ]^2))))
cat(sprintf("met_S (should be ~0.2): mean=%.6f, range=[%.6f, %.6f]\n",
            mean(rowSums(W2[met_S, ]^2)), min(rowSums(W2[met_S, ]^2)), max(rowSums(W2[met_S, ]^2))))
cat(sprintf("met_N (should be ~0): mean=%.6f, range=[%.6f, %.6f]\n",
            mean(rowSums(W2[met_N, ]^2)), min(rowSums(W2[met_N, ]^2)), max(rowSums(W2[met_N, ]^2))))

cat("Group-wise scaling complete\n")



# ============================================================
# DEFINE THE REMAINING NOISE IN THE VARIANCE PER METABOLITE
# ============================================================

#### Define the noise term in the equation
###### The main problem was that the noise term was overtaking the signal
###### That is why we need to scale the W matrices to achieve a target signal variance


# cat("\n=== SCALING TO TARGET SIGNAL VARIANCE ===\n")

# ## ---------- 1) Compute signal variance after TARGET scaling ----------
# # Signal variance per metabolite from scaled W1, W2, Ws
# signal_var_before <- rowSums(W1^2) + rowSums(W2^2) + rowSums(Ws^2)

# cat(sprintf("Signal variance before global scaling: [%.6f, %.6f]\n", 
#             min(signal_var_before), max(signal_var_before)))
# cat(sprintf("Mean signal variance before: %.6f\n", mean(signal_var_before)))

# ## ---------- 2) Global scaling to achieve target signal fraction ----------
# target_signal_fraction <- 0.95  # 80% signal, 20% noise
# target_total_var <- 1.0         # Target total variance per metabolite

# mean_signal_before <- mean(signal_var_before)
# if (mean_signal_before < 1e-12) {
#   stop("Signal variance is essentially zero - check W matrix construction")
# }

# # We want: mean(signal_var_after) = target_signal_fraction * target_total_var
# global_scale <- sqrt((target_signal_fraction * target_total_var) / mean_signal_before)

# cat(sprintf("Applying global scale factor: %.4f\n", global_scale))

# # Scale all weight matrices
# W1 <- W1 * global_scale
# W2 <- W2 * global_scale
# Ws <- Ws * global_scale

## ---------- 3) Compute final signal and noise variance ----------
cat("\n=== COMPUTING NOISE VARIANCE ===\n")
target_total_var <- 1.0 # Target total variance per metabolite
signal_var <- rowSums(W1^2) + rowSums(W2^2) + rowSums(Ws^2)

# Check if any metabolites exceed target total variance
n_exceed <- sum(signal_var > target_total_var)
if (n_exceed > 0) {
  cat(sprintf("WARNING: %d metabolites have signal variance > %.2f\n", 
              n_exceed, target_total_var))
  cat(sprintf("Max signal variance: %.6f\n", max(signal_var)))
  
  # scale down W matrices for metabolites that exceed
  idx_exceed <- which(signal_var > target_total_var)
  for (m in idx_exceed) {
    shrink_factor <- sqrt((target_total_var - 1e-6) / signal_var[m])
    W1[m, ] <- W1[m, ] * shrink_factor
    W2[m, ] <- W2[m, ] * shrink_factor
    Ws[m, ] <- Ws[m, ] * shrink_factor
  }
  
  signal_var <- rowSums(W1^2) + rowSums(W2^2) + rowSums(Ws^2)
  cat(sprintf("After capping, max signal variance: %.6f\n", max(signal_var)))
}

# # Compute noise variance as remainder
noise_var <- target_total_var - signal_var

# DIAGNOSTIC: Check W2 variance BEFORE minimum noise adjustment
cat("\n=== W2 DIAGNOSTIC (before minimum noise adjustment) ===\n")
cat(sprintf("met_B: mean=%.6f\n", mean(rowSums(W2[met_B, ]^2))))
cat(sprintf("met_V: mean=%.6f\n", mean(rowSums(W2[met_V, ]^2))))
cat(sprintf("met_S: mean=%.6f\n", mean(rowSums(W2[met_S, ]^2))))
cat(sprintf("ALL metabolites: sum=%.6f, max=%.6f\n", sum(rowSums(W2^2)), max(rowSums(W2^2))))

# Ensure non-negative and numerical stability
noise_var[noise_var < 0] <- 1e-6
noise_sd <- sqrt(noise_var)


## 4) Report results 
cat(sprintf("\nAfter global scaling:\n"))
cat(sprintf("  Signal variance: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(signal_var), min(signal_var), max(signal_var)))
cat(sprintf("  Noise variance: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(noise_var), min(noise_var), max(noise_var)))
cat(sprintf("  Signal-to-noise ratio: mean=%.2f, range=[%.2f, %.2f]\n",
            mean(signal_var / noise_var), 
            min(signal_var / noise_var), 
            max(signal_var / noise_var)))

##  5) Verify total variance 
total_var_check <- signal_var + noise_var
cat(sprintf("\nTotal variance check:\n"))
cat(sprintf("  Range: [%.6f, %.6f] (target: %.2f)\n", 
            min(total_var_check), max(total_var_check), target_total_var))
cat(sprintf("  All within tolerance: %s\n", 
            all(abs(total_var_check - target_total_var) < 1e-6)))

##  6) Check variance by metabolite group 
cat("\n=== Variance by Metabolite Group ===\n")

group_stats <- data.frame(
  group = c("met_B", "met_V", "met_S", "met_N"),
  mean_signal = c(mean(signal_var[1:30]), 
                  mean(signal_var[31:60]), 
                  mean(signal_var[61:90]), 
                  mean(signal_var[91:100])),
  mean_noise = c(mean(noise_var[1:30]), 
                 mean(noise_var[31:60]), 
                 mean(noise_var[61:90]), 
                 mean(noise_var[91:100]))
)
print(group_stats)

cat("\n=== FINAL GROUND TRUTH VARIANCE SHARES ===\n")

total_var <- signal_var + noise_var

# ============================================================================
# ANALYTICALLY EXACT GROUND TRUTH (Linear SCM)
# ============================================================================
# Since Y = W1·Z1_s + W2·Z2_s + Ws·Zc + ε with independent Z's and ε:
#
# Var(Y_i) = Σ_j W1[i,j]² · Var(Z1_s[j]) + Σ_k W2[i,k]² · Var(Z2_s[k])
#          + Σ_m Ws[i,m]² · Var(Zc[m]) + Var(ε_i)
#
# Since Var(Z) = 1 for all latents (sampled from N(0,1)):
# Var(Y_i) = ||W1[i,:]||² + ||W2[i,:]||² + ||Ws[i,:]||² + noise_var[i]
#
# This matches the data generation exactly (signal_var and noise_var computation above)
# ============================================================================

gt_shares <- list(
  Z1    = rowSums(W1^2) / total_var,
  Z2    = rowSums(W2^2) / total_var,
  Zs    = rowSums(Ws^2) / total_var,
  noise = noise_var / total_var
)

gt_df <- data.frame(
  met         = paste0("met_", 1:N_met),
  share_Z1    = gt_shares$Z1,
  share_Z2    = gt_shares$Z2,
  share_Zs    = gt_shares$Zs,
  share_noise = gt_shares$noise
)

gt_df$group <- "N"
gt_df$group[1:10] <- "B"
gt_df$group[11:20] <- "V"
gt_df$group[21:30] <- "S"

cat("\nGround truth shares by group (final):\n")
print(aggregate(cbind(share_Z1, share_Z2, share_Zs, share_noise) ~ group,
                data = gt_df, FUN = mean))


#Compositionality helper: softmax function
softmax <- function(x) {
  x_max <- max(x)
  exp_x <- exp(x - x_max)
  return(exp_x / sum(exp_x))
}




################################################################################
# CREATE THE HELPER FUNCTIONS FOR THE NONLINEAR MAPPING
################################################################################
# Following LEAF's approach: shallow MLP for nonlinear latent-to-observation mapping
#
# Both linear and nonlinear models output log-concentrations
# softmax(B_star) gives compositional closure

use_nonlinear_X <- FALSE 

# Initialize MLP weights (only needed for nonlinear mapping)
if (use_nonlinear_X) {
  cat("\n=== INITIALIZING MLP WEIGHTS ===\n")

  # Hidden layer dimensions (LEAF uses 100)
  h_bac <- 100L
  h_vir <- 100L

  cat(sprintf("Bacteria MLP: [%d input] → [%d hidden] → [%d output]\n",
              dim_Z1_s + dim_Zc, h_bac, N_bac))
  cat(sprintf("Virus MLP: [%d input] → [%d hidden] → [%d output]\n",
              dim_Z2_s + dim_Zc, h_vir, N_vir))

  # ===================================================================
  # BACTERIA MLP WEIGHTS
  # ===================================================================
  # Input: [Z1_s, Zc] (bacteria-specific + shared latents)
  # Output: log-concentrations for N_bac bacteria

  # First layer: input → hidden (with Xavier initialization)
  W1_bac <- matrix(
    rnorm((dim_Z1_s + dim_Zc) * h_bac, sd = 1 / sqrt(dim_Z1_s + dim_Zc)),
    nrow = dim_Z1_s + dim_Zc,
    ncol = h_bac
  )

  # Second layer: hidden → log-concentrations
  W2_bac <- matrix(
    rnorm(h_bac * N_bac, sd = 1 / sqrt(h_bac)),
    nrow = h_bac,
    ncol = N_bac
  )

  # ===================================================================
  # VIRUS MLP WEIGHTS
  # ===================================================================
  # Input: [Z2_s, Zc] (virus-specific + shared latents)
  # Output: log-concentrations for N_vir viruses

  # First layer: input → hidden (with Xavier initialization)
  W1_vir <- matrix(
    rnorm((dim_Z2_s + dim_Zc) * h_vir, sd = 1 / sqrt(dim_Z2_s + dim_Zc)),
    nrow = dim_Z2_s + dim_Zc,
    ncol = h_vir
  )

  # Second layer: hidden → log-concentrations
  W2_vir <- matrix(
    rnorm(h_vir * N_vir, sd = 1 / sqrt(h_vir)),
    nrow = h_vir,
    ncol = N_vir
  )

  cat("MLP weights initialized successfully\n")
}

compute_BV <- function(Z1_s, Z2_s, Zc,
                       use_nonlinear_X,
                       A1_s, A2_s, As1_shared, As2_shared,
                       W1_bac = NULL, W2_bac = NULL,
                       W1_vir = NULL, W2_vir = NULL) {
  if (!use_nonlinear_X) {
    # ===================================================================
    # LINEAR MODEL: Additive in log-concentration space
    # ===================================================================
    # B_star, V_star = log-concentrations (linear combination of latents)
    # Interpretation: log(concentration_i) = A_i·Z (additive model)
    B_star <- as.numeric(A1_s %*% Z1_s + As1_shared %*% Zc)
    V_star <- as.numeric(A2_s %*% Z2_s + As2_shared %*% Zc)
  } else {
    # ===================================================================
    # NONLINEAR MODEL: MLP in log-concentration space (LEAF-style)
    # ===================================================================
    # B_star, V_star = log-concentrations (nonlinear function of latents)
    # Interpretation: log(concentration_i) = MLP(Z)
    #
    # Architecture: 2-layer MLP with tanh activation
    # - Layer 1: [Z_private, Zc] → tanh → hidden (h_bac or h_vir dims)
    # - Layer 2: hidden → linear → log-concentrations (N_bac or N_vir dims)

    # Bacteria MLP
    z_cat_bac <- c(Z1_s, Zc)                         # Concatenate bacteria-specific + shared latents
    h_bac_vec <- tanh(drop(z_cat_bac %*% W1_bac))    # Hidden layer with tanh activation
    B_star    <- drop(h_bac_vec %*% W2_bac)          # Output: log-concentrations for bacteria

    # Virus MLP
    z_cat_vir <- c(Z2_s, Zc)                         # Concatenate virus-specific + shared latents
    h_vir_vec <- tanh(drop(z_cat_vir %*% W1_vir))    # Hidden layer with tanh activation
    V_star    <- drop(h_vir_vec %*% W2_vir)          # Output: log-concentrations for viruses
  }

  list(B_star = B_star, V_star = V_star)
}

# single simulation loop
cat("\n=== STARTING SIMULATION LOOP ===\n")
cat(sprintf("Running %d simulations...\n", n_sims))

# Print detailed info for first simulation only
print_first_sim <- TRUE



for (sim in 1:n_sims) {
  # =================================================================
  # SAMPLE LATENT FACTORS (all independent)
  # =================================================================
  Zc <- rnorm(dim_Zc)
  Z1_s <- rnorm(dim_Z1_s)
  Z2_s <- rnorm(dim_Z2_s)

  # Store latents
  Z1_store[sim, ] <- Z1_s
  Z2_store[sim, ] <- Z2_s
  Zs_store[sim, ] <- Zc

  # =================================================================
  # GENERATE X1, X2: Latent factors → Log-concentrations → Compositions
  # =================================================================
  # Pipeline (works for both linear and nonlinear models):
  # 1. Z (latents) → B_star/V_star (log-concentrations)
  #    - Linear: B_star = A·Z (additive)
  #    - Nonlinear: B_star = MLP(Z) (LEAF-style shallow MLP)
  # 2. Softmax closure: B_comp = exp(B_star) / sum(exp(B_star))
  #    - Converts log-concentrations → relative abundances (compositions)
  #    - Enforces simplex constraint (sum to 1)

  BV <- compute_BV(
    Z1_s = Z1_s,
    Z2_s = Z2_s,
    Zc   = Zc,
    use_nonlinear_X = use_nonlinear_X,
    A1_s = A1_s,
    A2_s = A2_s,
    As1_shared = As1_shared,
    As2_shared = As2_shared,
    W1_bac = if (use_nonlinear_X) W1_bac else NULL,
    W2_bac = if (use_nonlinear_X) W2_bac else NULL,
    W1_vir = if (use_nonlinear_X) W1_vir else NULL,
    W2_vir = if (use_nonlinear_X) W2_vir else NULL
  )

  # Extract log-concentrations
  B_star <- BV$B_star  # -concentrations for bacteria
  V_star <- BV$V_star  # concentrations for viruses






  # log_B_star <- as.numeric(A1_s %*% Z1_s + As_shared %*% Zc) # As_shared for X1 = X2
  # log_V_star <- as.numeric(A2_s %*% Z2_s + As_shared %*% Zc) # As_shared for X1 = X2

  #B_star <- as.numeric(A1_s %*% Z1_s + As1_shared %*% Zc) # This is for the COPSAC clone
  #V_star <- as.numeric(A2_s %*% Z2_s + As2_shared %*% Zc) #this is for the COPSAC clone

  # =================================================================
  # APPLY SPARSITY: Zero out low-abundance taxa before compositional closure
  # =================================================================
  # Convert log-concentrations to absolute concentrations
  #B_abs <- exp(B_star)
  #V_abs <- exp(V_star)

  # Apply detection limit (taxa below threshold are considered absent)
  #B_abs[B_abs < detection_limit] <- 00
#  3V_abs[V_abs < detection_limit] <- 0

  # Apply compositional closure (softmax on log scale, accounting for zeros)
  # For zero taxa, we keep them at zero; for non-zero, apply softmax


  B_star <- BV$B_star  # Log-concentrations for bacteria
  V_star <- BV$V_star  # Log-concentrations for viruses

  # Apply compositional closure (softmax)
  B_comp <- softmax(B_star)  # Bacterial relative abundances (sum to 1)
  V_comp <- softmax(V_star)  # Viral relative abundances (sum to 1)

  # =================================================================
  # SPARSITY DIAGNOSTICS: Calculate percentage of zeros
  # =================================================================
 # sparsity_B[sim] <- 100 * sum(B_comp == 0) / length(B_comp)
 # sparsity_V[sim] <- 100 * sum(V_comp == 0) / length(V_comp)



  # Generate Y: Y = W1*Z1_s + W2*Z2_s + Ws*Zc + noise

  eps_y <- rnorm(N_met, mean = 0, sd = noise_sd)
  Y_log <- as.numeric(W1 %*% Z1_s + W2 %*% Z2_s + Ws %*% Zc) + eps_y

# Store results
Y_mat[sim, ] <- Y_log
X1_mat[sim, ] <- B_comp   # Bacterial abundances
X2_mat[sim, ] <- V_comp   # Viral abundances
}


################################################################################
# SPARSITY DIAGNOSTICS REPORT
################################################################################
cat("\n=== SPARSITY DIAGNOSTICS REPORT ===\n")
cat(sprintf("Detection limit applied: %.2e\n", detection_limit))
cat("\n--- Bacteria (B_comp) Sparsity ---\n")
cat(sprintf("  Mean percentage of zeros: %.2f%%\n", mean(sparsity_B)))
cat(sprintf("  Median percentage of zeros: %.2f%%\n", median(sparsity_B)))
cat(sprintf("  Range: [%.2f%%, %.2f%%]\n", min(sparsity_B), max(sparsity_B)))
cat(sprintf("  SD: %.2f%%\n", sd(sparsity_B)))

cat("\n--- Viruses (V_comp) Sparsity ---\n")
cat(sprintf("  Mean percentage of zeros: %.2f%%\n", mean(sparsity_V)))
cat(sprintf("  Median percentage of zeros: %.2f%%\n", median(sparsity_V)))
cat(sprintf("  Range: [%.2f%%, %.2f%%]\n", min(sparsity_V), max(sparsity_V)))
cat(sprintf("  SD: %.2f%%\n", sd(sparsity_V)))

# Create sparsity data frame for export
sparsity_df <- data.frame(
  sim = 1:n_sims,
  bacteria_pct_zeros = sparsity_B,
  viruses_pct_zeros = sparsity_V
)

# Save sparsity diagnostics
write.csv(sparsity_df, "sparsity_diagnostics_complex.csv", row.names = FALSE)
cat("\nSparsity diagnostics saved to 'sparsity_diagnostics_complex.csv'\n")

################################################################################
# VERIFY EMPIRICAL VARIANCE MATCHES GROUND TRUTH
################################################################################
cat("\n=== VERIFYING EMPIRICAL VS THEORETICAL VARIANCE ===\n")

# Compute empirical variance from generated Y_mat
empirical_total_var <- apply(Y_mat, 2, var)

# Theoretical total variance from ground truth
theoretical_total_var <- signal_var + noise_var

# Compare
cat(sprintf("Empirical total variance: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(empirical_total_var), min(empirical_total_var), max(empirical_total_var)))
cat(sprintf("Theoretical total variance: mean=%.6f, range=[%.6f, %.6f]\n",
            mean(theoretical_total_var), min(theoretical_total_var), max(theoretical_total_var)))

# Check if they match
variance_ratio <- empirical_total_var / theoretical_total_var
cat(sprintf("\nEmpirical / Theoretical ratio: mean=%.4f, range=[%.4f, %.4f]\n",
            mean(variance_ratio), min(variance_ratio), max(variance_ratio)))

if (any(abs(variance_ratio - 1.0) > 0.1)) {
  warning("MISMATCH: Empirical variance differs from theoretical by >10%!")
  cat("This suggests a bug in data generation or ground truth calculation.\n")
} else {
  cat("SUCCESS: Empirical variance matches theoretical variance!\n")
}



# # =================================================================
# # CLR TRANSFORMATION (after simulation loop)
# # =================================================================

# clr_transform <- function(x, pseudocount = 1e-6) {
#   x <- as.matrix(x)
#   x_pseudo <- x + pseudocount

#   # compositions::clr works on acomp objects.
#   # It treats rows as observations.
#   clr_mat <- compositions::clr(compositions::acomp(x_pseudo))

#   return(as.matrix(clr_mat))
# }


# X1_clr <- clr_transform(X1_mat)
# X2_clr <- clr_transform(X2_mat)

# X1_mat <- X1_clr
# X2_mat <- X2_clr
# cat("\n=== CLR TRANSFORMATION COMPLETE ===\n")
# cat(sprintf("X1_clr range: [%.4f, %.4f]\n", min(X1_clr), max(X1_clr)))
# cat(sprintf("X2_clr range: [%.4f, %.4f]\n", min(X2_clr), max(X2_clr)))


################################################################################
# BALANCED SBP ILR ON SYNTHETIC X1_mat, X2_mat
################################################################################

# # 1) Build balanced SBP for a given set of features
# sbp_balanced <- function(features) {
#   D <- length(features)
#   if (D < 2) stop(sprintf("Need at least 2 features for SBP, got D = %d", D))

#   splits <- list()
#   q <- list(seq_len(D))  # queue of index vectors

#   while (length(q) > 0) {
#     grp <- q[[1]]; q <- q[-1]
#     if (length(grp) <= 1) next
#     k <- length(grp)
#     k_left <- floor(k / 2)
#     left  <- grp[seq_len(k_left)]
#     right <- grp[(k_left + 1):k]
#     splits[[length(splits) + 1]] <- list(num = left, den = right)
#     q[[length(q) + 1]] <- left
#     q[[length(q) + 1]] <- right
#   }

#   if (length(splits) != D - 1) {
#     stop(sprintf("Balanced SBP generated %d splits, expected %d.", length(splits), D - 1))
#   }

#   sbp <- matrix(0L, nrow = D, ncol = D - 1,
#                 dimnames = list(features, paste0("split", seq_len(D - 1))))
#   for (j in seq_along(splits)) {
#     sbp[splits[[j]]$num, j] <-  1L
#     sbp[splits[[j]]$den, j] <- -1L
#   }
#   sbp
# }


# # 2) Convert SBP to orthonormal ILR basis
# sbp_to_basis <- function(sbp) {
#   sbp <- as.matrix(sbp)
#   D <- nrow(sbp); K <- ncol(sbp)
#   if (K != D - 1) stop("SBP must have D rows and D-1 columns.")
#   B <- matrix(0, nrow = D, ncol = K)
#   for (j in seq_len(K)) {
#     r <- sum(sbp[, j] ==  1L)
#     s <- sum(sbp[, j] == -1L)
#     if (r == 0 || s == 0) stop(sprintf("Split %d must have both +1 and -1 groups.", j))
#     B[sbp[, j] ==  1L, j] <-  sqrt(s / (r * (r + s)))
#     B[sbp[, j] == -1L, j] <- -sqrt(r / (s * (r + s)))
#   }
#   B
# }

# # 3) ILR using that basis
# ilr_transform <- function(M, sbp, pseudocount = 1e-6) {
#   M <- as.matrix(M)  # samples in rows, features in columns
#   M_safe <- M + pseudocount
#   D <- ncol(M_safe)
#   if (D < 2) stop("Need at least 2 features for ILR.")

#   if (nrow(sbp) != D || ncol(sbp) != D - 1)
#     stop("SBP must be D x (D-1) with D = ncol(M).")

#   B <- sbp_to_basis(sbp)
#   rownames(B) <- colnames(M_safe)

#   L   <- log(M_safe)
#   clr <- sweep(L, 1, rowMeans(L), "-")  # row-wise centering
#   Z <- clr %*% B
#   rownames(Z) <- rownames(M_safe)
#   colnames(Z) <- colnames(B)
#   Z
# }

# ################################################################################
# # APPLY BALANCED ILR TO SYNTHETIC X1_mat, X2_mat
# ################################################################################

# cat("\n=== ILR TRANSFORMATION WITH BALANCED SBP ON SYNTHETIC DATA ===\n")

# # Ensure feature names exist, otherwise SBP will fail
# if (is.null(colnames(X1_mat))) {
#   colnames(X1_mat) <- paste0("bac_", seq_len(ncol(X1_mat)))
# }
# if (is.null(colnames(X2_mat))) {
#   colnames(X2_mat) <- paste0("vir_", seq_len(ncol(X2_mat)))
# }

# # Build SBP and transform
# sbp_bac <- sbp_balanced(colnames(X1_mat))
# X1_ilr  <- ilr_transform(X1_mat, sbp = sbp_bac)

# sbp_vir <- sbp_balanced(colnames(X2_mat))
# X2_ilr  <- ilr_transform(X2_mat, sbp = sbp_vir)

# # Overwrite for downstream LEAF
# X1_mat <- X1_ilr
# X2_mat <- X2_ilr

# cat("=== ILR TRANSFORMATION COMPLETE ===\n")
# cat(sprintf("X1_ilr dimensions: %d samples x %d ILR coords\n", nrow(X1_mat), ncol(X1_mat)))
# cat(sprintf("X2_ilr dimensions: %d samples x %d ILR coords\n", nrow(X2_mat), ncol(X2_mat)))




################################################################################
# LATENT FACTOR CORRELATION ANALYSIS
################################################################################
# Check disentanglement: Z1_s and Z2_s should be independent (uncorrelated)
# Zc should be independent of both Z1_s and Z2_s in the generative model
################################################################################
cat("\n=== ANALYZING LATENT FACTOR CORRELATIONS ===\n")

# Calculate correlations between stored latent factors across simulations
# For each pair of latent dimensions, compute correlation

# Average correlation between Z1_s and Z2_s dimensions
cor_Z1_Z2 <- cor(Z1_store, Z2_store)
mean_abs_cor_Z1_Z2 <- mean(abs(cor_Z1_Z2))
max_abs_cor_Z1_Z2 <- max(abs(cor_Z1_Z2))

# Average correlation between Zc and Z1_s dimensions
cor_Zc_Z1 <- cor(Zs_store, Z1_store)
mean_abs_cor_Zc_Z1 <- mean(abs(cor_Zc_Z1))
max_abs_cor_Zc_Z1 <- max(abs(cor_Zc_Z1))

# Average correlation between Zc and Z2_s dimensions
cor_Zc_Z2 <- cor(Zs_store, Z2_store)
mean_abs_cor_Zc_Z2 <- mean(abs(cor_Zc_Z2))
max_abs_cor_Zc_Z2 <- max(abs(cor_Zc_Z2))

cat("\n--- Correlation Summary (should be near 0 for independent sampling) ---\n")
cat(sprintf("Z1_s vs Z2_s:\n"))
cat(sprintf("  Mean absolute correlation: %.6f\n", mean_abs_cor_Z1_Z2))
cat(sprintf("  Max absolute correlation:  %.6f\n", max_abs_cor_Z1_Z2))

cat(sprintf("\nZc vs Z1_s:\n"))
cat(sprintf("  Mean absolute correlation: %.6f\n", mean_abs_cor_Zc_Z1))
cat(sprintf("  Max absolute correlation:  %.6f\n", max_abs_cor_Zc_Z1))

cat(sprintf("\nZc vs Z2_s:\n"))
cat(sprintf("  Mean absolute correlation: %.6f\n", mean_abs_cor_Zc_Z2))
cat(sprintf("  Max absolute correlation:  %.6f\n", max_abs_cor_Zc_Z2))

cat("\nInterpretation:\n")
cat("- Values near 0: Good disentanglement (latents are independent)\n")
cat("- Values > 0.3: Potential leakage between latent factors\n")
cat("- Expected: All correlations should be ~0 since we sample independently\n")

# # If Zc is truly shared, X1 and X2 should be correlated
# # because they both contain t(Tc) %*% Zc

# cat("\n=== VERIFYING SHARED INFORMATION ===\n")

# # Correlation between X1 and X2 features
# cor_mat <- cor(X1_mat, X2_mat)
# cat(sprintf("Mean |cor(X1[i], X2[j])|: %.4f\n", mean(abs(cor_mat))))

# # Diagonal should be especially high if N_bac == N_vir
# # because X1[i] and X2[i] share the same Tc[,i] column
# cat(sprintf("Mean |cor(X1[i], X2[i])| (diagonal): %.4f\n", mean(abs(diag(cor_mat)))))

# NOTE: Data frames will be created AFTER relative abundance computation
# to ensure we save the transformed data, not the compositional data
# See lines after ground truth calculation section

# make_virome_gt_linear <- function(
#   W1, W2, Ws,
#   z1_var   = 1.0,
#   z2_var   = 1.0,
#   zs_var   = 1.0,
#   noise_var  # length N_met, variance of eps_y per metabolite
# ) {
#   N_met <- nrow(W1)
#   shares <- list(
#     Z1    = numeric(N_met),
#     Z2    = numeric(N_met),
#     Zs    = numeric(N_met),
#     noise = numeric(N_met)
#   )

#   for (m in 1:N_met) {
#     num_Z1  <- sum(W1[m, ]^2) * z1_var
#     num_Z2  <- sum(W2[m, ]^2) * z2_var
#     num_Zs  <- sum(Ws[m, ]^2) * zs_var
#     num_eps <- noise_var[m]

#     denom <- num_Z1 + num_Z2 + num_Zs + num_eps
#     if (denom <= 0) {
#       shares$noise[m] <- 1
#     } else {
#       shares$Z1[m]    <- num_Z1  / denom
#       shares$Z2[m]    <- num_Z2  / denom
#       shares$Zs[m]    <- num_Zs  / denom
#       shares$noise[m] <- num_eps / denom
#     }
#   }

#   shares
# }



# gt_shares <- make_virome_gt_linear(
#   W1 = W1,
#   W2 = W2,
#   Ws = Ws,
#   z1_var = 1.0,
#   z2_var = 1.0,
#   zs_var = 1.0,
#   noise_var = noise_var # 1- signal_var
# )


# gt_df <- data.frame(
#   met         = paste0("met_", 1:ncol(Y_mat)),
#   share_Z1    = gt_shares$Z1,
#   share_Z2    = gt_shares$Z2,
#   share_Zs    = gt_shares$Zs,
#   share_noise = gt_shares$noise
# )

# summary(gt_df$share_noise)
# summary(gt_df$share_Z1)
# summary(gt_df$share_Z2)
# summary(gt_df$share_Zs)


################################################################################
# METABOLITE CLASSIFICATION BY DOMINANT VARIANCE SOURCE
################################################################################
# - "mixed": No clear dominant source
################################################################################

## ---------- assign metabolite groups based on dominant variance source ----------

# Dominance threshold
dom_thr <- 0.35

# start with all metabolites as "mixed"
gt_df$group <- "mixed"

# Z1 dominated (bacteria driven)
gt_df$group[
  gt_df$share_Z1 >= dom_thr &
  gt_df$share_Z1 >= gt_df$share_Z2 &
  gt_df$share_Z1 >= gt_df$share_Zs &
  gt_df$share_Z1 >= gt_df$share_noise
] <- "Z1_dominant"

# Z2 dominated (virus driven via AMGs)
gt_df$group[
  gt_df$share_Z2 >= dom_thr &
  gt_df$share_Z2 >= gt_df$share_Z1 &
  gt_df$share_Z2 >= gt_df$share_Zs &
  gt_df$share_Z2 >= gt_df$share_noise
] <- "Z2_dominant"

# shared Zs dominated
gt_df$group[
  gt_df$share_Zs >= dom_thr &
  gt_df$share_Zs >= gt_df$share_Z1 &
  gt_df$share_Zs >= gt_df$share_Z2 &
  gt_df$share_Zs >= gt_df$share_noise
] <- "Zs_dominant"

# noise dominated (essentially uninformative metabolites)
gt_df$group[
  gt_df$share_noise >= dom_thr &
  gt_df$share_noise >= gt_df$share_Z1 &
  gt_df$share_noise >= gt_df$share_Z2 &
  gt_df$share_noise >= gt_df$share_Zs
] <- "noise_dominant"

# inspect how many in each group
table(gt_df$group)

gt_df <-gt_df %>%
  select(-group)

write.csv(gt_df, "GT_virome_variance_shares_complex_COPSAC.csv", row.names = FALSE)


Z1_df <- data.frame(sim = 1:n_sims, Z1_store)
Z2_df <- data.frame(sim = 1:n_sims, Z2_store)
Zs_df <- data.frame(sim = 1:n_sims, Zs_store)

write.csv(Z1_df, "Z1_latents_RA_complex_COPSAC.csv", row.names = FALSE)
write.csv(Z2_df, "Z2_latents_RA_complex_COPSAC.csv", row.names = FALSE)
write.csv(Zs_df, "Zs_latents_RA_complex_COPSAC.csv", row.names = FALSE)


###################################################################




################################################################################
# BUILD FINAL OUTPUT DATA FRAMES (AFTER RELATIVE ABUNDANCE COMPUTATION)
################################################################################
cat("\n=== BUILDING FINAL OUTPUT DATA FRAMES ===\n")

# Create ID columns
sim_ids <- 1:n_sims
rep_ids <- rep(1L, n_sims)

# Build final data frames with relative abundance data
# Y_mat is still in log space (metabolites)
# X1_mat and X2_mat are now relative abundances (sum-to-one constraint)
Y_df  <- data.frame(sim = sim_ids, rep = rep_ids, Y_mat)
X1_df <- data.frame(sim = sim_ids, rep = rep_ids, X1_mat)  # relative abundance bacteria
X2_df <- data.frame(sim = sim_ids, rep = rep_ids, X2_mat)  # relative abundance viruses

# Set appropriate column names
colnames(Y_df)[-(1:2)]  <- paste0("met_", 1:ncol(Y_mat))
colnames(X1_df)[-(1:2)] <- paste0("cbac_", 1:ncol(X1_mat))
colnames(X2_df)[-(1:2)] <- paste0("vir_", 1:ncol(X2_mat))

cat(sprintf("Y_df: %d samples x %d metabolites\n", nrow(Y_df), ncol(Y_df) - 2))
cat(sprintf("X1_df: %d samples x %d relative abundances (bacteria)\n", nrow(X1_df), ncol(X1_df) - 2))
cat(sprintf("X2_df: %d samples x %d relative abundances (viruses)\n", nrow(X2_df), ncol(X2_df) - 2))

write.csv(Y_df,  file = "Y_metabolites_log_synthetic_complex_RA_COPSAC.csv", row.names = FALSE)
write.csv(X1_df, file = "X1_bacteria_synthetic_RA_complex_COPSAC.csv",  row.names = FALSE)
write.csv(X2_df, file = "X2_viruses_synthetic_RA_complex_COPSAC.csv",   row.names = FALSE)


# # ---------------------------
# # 3) Save to CSV
# # ---------------------------

# #write.csv(Y_df, file = "X2_viruses_last_t.csv", row.names = FALSE)

# #write.csv(cij_x1_df, file = "Cij_X1_to_Y.csv", row.names = FALSE)
# #write.csv(cij_x2_df, file = "Cij_X2_to_Y.csv", row.names = FALSE)


# ################################################################################
# # VALIDATION PLOTS
# ################################################################################
# cat("\n=== GENERATING VALIDATION PLOTS ===\n")

# # Load required libraries
# library(ggplot2)
# library(tidyr)
# library(dplyr)
# library(viridis)

# # Create output directory
# dir.create("validation_plots", showWarnings = FALSE)

# ################################################################################
# # PLOT 2: LATENT FACTOR CORRELATION HEATMAP
# ################################################################################
# cat("  Creating Plot 2: Latent correlation heatmap...\n")

# # Combine all latents into one matrix
# all_latents <- cbind(Z1_store, Z2_store, Zs_store)
# colnames(all_latents) <- c(
#   paste0("Z1_", 1:dim_Z1_s),
#   paste0("Z2_", 1:dim_Z2_s),
#   paste0("Zc_", 1:dim_Zc)
# )

# # Compute correlation matrix
# cor_matrix <- cor(all_latents)

# # Convert to long format for ggplot
# cor_df <- as.data.frame(as.table(cor_matrix))
# colnames(cor_df) <- c("Var1", "Var2", "value")

# # Compute mean correlations for subtitle
# z1_indices <- 1:dim_Z1_s
# z2_indices <- (dim_Z1_s + 1):(dim_Z1_s + dim_Z2_s)
# zc_indices <- (dim_Z1_s + dim_Z2_s + 1):(dim_Z1_s + dim_Z2_s + dim_Zc)

# mean_z1_z2 <- mean(abs(cor_matrix[z1_indices, z2_indices]))
# mean_z1_zc <- mean(abs(cor_matrix[z1_indices, zc_indices]))
# mean_z2_zc <- mean(abs(cor_matrix[z2_indices, zc_indices]))

# subtitle_text <- sprintf(
#   "Mean |cor|: Z1-Z2 = %.3f, Z1-Zc = %.3f, Z2-Zc = %.3f (should be ~0)",
#   mean_z1_z2, mean_z1_zc, mean_z2_zc
# )

# # Create heatmap
# p2 <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = value)) +
#   geom_tile() +
#   scale_fill_gradient2(
#     low = "#2166ac",
#     mid = "white",
#     high = "#b2182b",
#     midpoint = 0,
#     limits = c(-1, 1),
#     name = "Correlation"
#   ) +
#   # Add dashed lines to separate Z1/Z2/Zc blocks
#   geom_vline(xintercept = dim_Z1_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.8) +
#   geom_vline(xintercept = dim_Z1_s + dim_Z2_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.8) +
#   geom_hline(yintercept = dim_Z1_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.8) +
#   geom_hline(yintercept = dim_Z1_s + dim_Z2_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.8) +
#   labs(
#     title = "Latent Factor Correlation Matrix",
#     subtitle = subtitle_text,
#     x = "Latent Factors",
#     y = "Latent Factors"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
#     axis.text.y = element_text(size = 8),
#     plot.title = element_text(face = "bold", hjust = 0.5),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30"),
#     panel.grid = element_blank()
#   )

# ggsave("validation_plots/02_latent_correlations.png", p2,
#        width = 10, height = 9, dpi = 300)
# cat("    Saved to validation_plots/02_latent_correlations.png\n")


# ################################################################################
# # PLOT 4: METABOLITE INTENSITY DISTRIBUTIONS (PER METABOLITE)
# ################################################################################
# cat("  Creating Plot 4: Metabolite intensity distributions...\n")

# # Read ground truth to get variance shares for coloring
# gt_temp <- read.csv("GT_virome_variance_shares_complex.csv")

# # Classify metabolites by dominant variance source
# dom_thr <- 0.35
# met_groups <- rep("Mixed", N_met)
# met_groups[gt_temp$share_Z1 >= dom_thr &
#            gt_temp$share_Z1 >= gt_temp$share_Z2 &
#            gt_temp$share_Z1 >= gt_temp$share_Zs &
#            gt_temp$share_Z1 >= gt_temp$share_noise] <- "Bacteria"

# met_groups[gt_temp$share_Z2 >= dom_thr &
#            gt_temp$share_Z2 >= gt_temp$share_Z1 &
#            gt_temp$share_Z2 >= gt_temp$share_Zs &
#            gt_temp$share_Z2 >= gt_temp$share_noise] <- "Virus"

# met_groups[gt_temp$share_Zs >= dom_thr &
#            gt_temp$share_Zs >= gt_temp$share_Z1 &
#            gt_temp$share_Zs >= gt_temp$share_Z2 &
#            gt_temp$share_Zs >= gt_temp$share_noise] <- "Shared"

# met_groups[gt_temp$share_noise >= dom_thr &
#            gt_temp$share_noise >= gt_temp$share_Z1 &
#            gt_temp$share_noise >= gt_temp$share_Z2 &
#            gt_temp$share_noise >= gt_temp$share_Zs] <- "Noise"

# # Create long-format data for plotting
# Y_long <- Y_df %>%
#   select(-sim, -rep) %>%
#   pivot_longer(cols = everything(), names_to = "metabolite", values_to = "abundance") %>%
#   mutate(
#     met_num = as.numeric(gsub("met_", "", metabolite)),
#     group = factor(met_groups[met_num], levels = c("Bacteria", "Virus", "Shared", "Noise", "Mixed"))
#   )

# # Create boxplot for each metabolite, colored by dominant source
# p4 <- ggplot(Y_long, aes(x = metabolite, y = abundance, fill = group)) +
#   geom_boxplot(alpha = 0.8, outlier.size = 0.3, outlier.alpha = 0.2) +
#   scale_fill_manual(
#     values = c(
#       "Bacteria" = "#4292c6",
#       "Virus" = "#d7301f",
#       "Shared" = "#9e9ac8",
#       "Noise" = "#969696",
#       "Mixed" = "#bdbdbd"
#     ),
#     name = "Dominant\nSource"
#   ) +
#   labs(
#     title = "Metabolite Intensity Distributions",
#     subtitle = sprintf("n = %d samples, %d metabolites (colored by dominant variance source)", n_sims, N_met),
#     x = "Metabolite",
#     y = "Abundance"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     plot.title = element_text(face = "bold", hjust = 0.5),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30"),
#     axis.text.x = element_blank(),
#     axis.ticks.x = element_blank(),
#     panel.grid.major.x = element_blank()
#   )

# ggsave("validation_plots/04_Y_metabolite_intensities.png", p4,
#        width = 14, height = 6, dpi = 300)
# cat("    Saved to validation_plots/04_Y_metabolite_intensities.png\n")


# ################################################################################
# # PLOT 5: VIRAL HOST RANGE DISTRIBUTION
# ################################################################################
# cat("  Creating Plot 5: Viral host range distribution...\n")

# # Calculate host range for each virus
# hosts_per_virus <- rowSums(Infection)

# # Create virus classification by selectivity
# virus_selectivity_class <- ifelse(hosts_per_virus == 1, "Specialist",
#                             ifelse(hosts_per_virus <= 10, "Moderate", "Generalist"))

# # Create data frame for plotting
# virus_host_df <- data.frame(
#   virus = 1:N_vir,
#   n_hosts = hosts_per_virus,
#   selectivity = factor(virus_selectivity_class, levels = c("Specialist", "Moderate", "Generalist"))
# )

# # Count viruses in each category
# selectivity_counts <- table(virus_selectivity_class)

# # Left panel: Histogram of host ranges
# p5a <- ggplot(virus_host_df, aes(x = n_hosts, fill = selectivity)) +
#   geom_histogram(binwidth = 2, alpha = 0.8, color = "white") +
#   scale_fill_manual(
#     values = c(
#       "Specialist" = "#d7301f",
#       "Moderate" = "#fdae6b",
#       "Generalist" = "#31a354"
#     ),
#     name = "Virus Type"
#   ) +
#   labs(
#     title = "Viral Host Range Distribution",
#     subtitle = sprintf("%d specialists, %d moderate, %d generalists",
#                       selectivity_counts["Specialist"],
#                       selectivity_counts["Moderate"],
#                       selectivity_counts["Generalist"]),
#     x = "Number of Bacterial Hosts",
#     y = "Count"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     plot.title = element_text(face = "bold", hjust = 0.5),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30")
#   )

# # # Right panel: Boxplot by selectivity class
# # p5b <- ggplot(virus_host_df, aes(x = selectivity, y = n_hosts, fill = selectivity)) +
# #   geom_boxplot(alpha = 0.7, outlier.alpha = 0.5) +
# #   scale_fill_manual(
# #     values = c(
# #       "Specialist" = "#d7301f",
# #       "Moderate" = "#fdae6b",
# #       "Generalist" = "#31a354"
# #     )
# #   ) +
# #   labs(
# #     title = "Hosts by Selectivity",
# #     x = "Virus Type",
# #     y = "Number of Hosts"
# #   ) +
# #   theme_minimal(base_size = 12) +
# #   theme(
# #     plot.title = element_text(face = "bold", hjust = 0.5),
# #     legend.position = "none"
# #   )

# # Combine plots side by side
# library(gridExtra)

# ggsave("validation_plots/05_viral_host_ranges.png", p5a,
#        width = 14, height = 6, dpi = 300)
# cat("    Saved to validation_plots/05_viral_host_ranges.png\n")


# ################################################################################
# # PLOT 6: THREE INTERACTION MATRICES (Cij, C_virus, Infection)
# ################################################################################
# cat("  Creating Plot 6: Three interaction matrices...\n")

# # Panel A: Cij (Bacteria-Metabolite interactions)
# cij_df <- expand.grid(
#   bacterium = 1:N_bac,
#   metabolite = 1:N_met
# )
# cij_df$value <- as.vector(Cij)
# cij_active <- cij_df %>% filter(value > 0)

# p6a <- ggplot(cij_active, aes(x = bacterium, y = metabolite)) +
#   geom_tile(aes(fill = value), color = NA) +
#   scale_fill_viridis_c(option = "plasma", name = "Consumption\nRate") +
#   scale_x_continuous(breaks = seq(0, N_bac, by = 20), expand = c(0, 0)) +
#   scale_y_continuous(breaks = seq(0, N_met, by = 20), expand = c(0, 0)) +
#   labs(
#     title = "Bacteria-Metabolite Interactions",
#     subtitle = sprintf("%.1f%% density | Bacteria consume metabolites",
#                       100 * nrow(cij_active) / (N_bac * N_met)),
#     x = "Bacterial Species",
#     y = "Metabolite"
#   ) +
#   theme_minimal(base_size = 10) +
#   theme(
#     plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30", size = 9),
#     panel.grid = element_blank(),
#     panel.border = element_rect(color = "gray50", fill = NA, linewidth = 1),
#     legend.position = "right"
#   )

# # Panel B: C_virus (Viral AMG-Metabolite interactions)
# cvirus_df <- expand.grid(
#   virus = 1:N_vir,
#   metabolite = 1:N_met
# )
# cvirus_df$value <- as.vector(C_virus)
# cvirus_active <- cvirus_df %>% filter(value > 0)

# p6b <- ggplot(cvirus_active, aes(x = virus, y = metabolite)) +
#   geom_tile(aes(fill = value), color = NA) +
#   scale_fill_viridis_c(option = "viridis", name = "AMG\nStrength") +
#   scale_x_continuous(breaks = seq(0, N_vir, by = 20), expand = c(0, 0)) +
#   scale_y_continuous(breaks = seq(0, N_met, by = 20), expand = c(0, 0)) +
#   labs(
#     title = "Viral AMG-Metabolite Interactions",
#     subtitle = sprintf("%.1f%% density | AMGs boost metabolism",
#                       100 * nrow(cvirus_active) / (N_vir * N_met)),
#     x = "Viral Phage",
#     y = "Metabolite"
#   ) +
#   theme_minimal(base_size = 10) +
#   theme(
#     plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30", size = 9),
#     panel.grid = element_blank(),
#     panel.border = element_rect(color = "gray50", fill = NA, linewidth = 1),
#     legend.position = "right"
#   )

# # Panel C: Infection (Virus-Bacteria network)
# infection_df <- expand.grid(
#   virus = 1:N_vir,
#   bacterium = 1:N_bac
# )
# infection_df$infects <- as.vector(t(Infection))
# infection_active <- infection_df %>% filter(infects == 1)

# p6c <- ggplot(infection_active, aes(x = bacterium, y = virus)) +
#   geom_tile(fill = "#4292c6", alpha = 0.7, color = NA) +
#   scale_x_continuous(breaks = seq(0, N_bac, by = 20), expand = c(0, 0)) +
#   scale_y_continuous(breaks = seq(0, N_vir, by = 20), expand = c(0, 0)) +
#   labs(
#     title = "Virus-Bacteria Infection Network (Infection)",
#     subtitle = sprintf("%.1f%% density | Viruses infect bacteria",
#                       100 * sum(Infection) / (N_vir * N_bac)),
#     x = "Bacterial Host",
#     y = "Viral Phage"
#   ) +
#   theme_minimal(base_size = 10) +
#   theme(
#     plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
#     plot.subtitle = element_text(hjust = 0.5, color = "gray30", size = 9),
#     panel.grid = element_blank(),
#     panel.border = element_rect(color = "gray50", fill = NA, linewidth = 1)
#   )

# # Combine all three plots
# library(gridExtra)
# p6_combined <- grid.arrange(p6a, p6b, p6c, ncol = 3)
# # Save three separate plots
# ggsave("validation_plots/06a_Cij_bacteria_metabolite.png", p6a,
#        width = 6, height = 6, dpi = 300)
# ggsave("validation_plots/06b_Cvirus_viralAMG_metabolite.png", p6b,
#        width = 6, height = 6, dpi = 300)
# ggsave("validation_plots/06c_Infection_virus_bacteria.png", p6c,
#        width = 6, height = 6, dpi = 300)

# cat("    Saved separate plots:\n")
# cat("      validation_plots/06a_Cij_bacteria_metabolite.png\n")
# cat("      validation_plots/06b_Cvirus_viralAMG_metabolite.png\n")
# cat("      validation_plots/06c_Infection_virus_bacteria.png\n")

# # Save combined plot (same as before)
# library(gridExtra)
# p6_combined <- grid.arrange(p6a, p6b, p6c, ncol = 3)

# ggsave("validation_plots/06_three_interaction_matrices.png", p6_combined,
#        width = 18, height = 6, dpi = 300)
# cat("    Saved merged plot to validation_plots/06_three_interaction_matrices.png\n")


# ################################################################################
# # SUMMARY STATISTICS CSV
# ################################################################################
# cat("  Creating summary statistics CSV...\n")

# # Calculate infection statistics
# total_infections <- sum(Infection)
# infection_density <- 100 * total_infections / (N_vir * N_bac)
# hosts_per_virus <- rowSums(Infection)
# mean_hosts <- mean(hosts_per_virus)
# sd_hosts <- sd(hosts_per_virus)

# summary_stats <- data.frame(
#   Metric = c(
#     "Number of simulations",
#     "Number of bacteria",
#     "Number of viruses",
#     "Number of metabolites",
#     "Z1 dimension (bacteria-specific)",
#     "Z2 dimension (virus-specific)",
#     "Zc dimension (shared)",
#     "Total infections",
#     "Infection density (%)",
#     "Mean hosts per virus",
#     "SD hosts per virus",
#     "Total AMGs",
#     "Mean |cor(Z1,Z2)|",
#     "Mean |cor(Z1,Zc)|",
#     "Mean |cor(Z2,Zc)|"
#   ),
#   Value = c(
#     n_sims,
#     N_bac,
#     N_vir,
#     N_met,
#     dim_Z1_s,
#     dim_Z2_s,
#     dim_Zc,
#     total_infections,
#     round(infection_density, 2),
#     round(mean_hosts, 2),
#     round(sd_hosts, 2),
#     sum(C_virus > 0),
#     round(mean_z1_z2, 4),
#     round(mean_z1_zc, 4),
#     round(mean_z2_zc, 4)
#   )
# )

# write.csv(summary_stats, "validation_plots/summary_statistics.csv", row.names = FALSE)
# cat("    Saved to validation_plots/summary_statistics.csv\n")

# cat("\n=== ALL VALIDATION PLOTS COMPLETE ===\n")
# cat("Plots saved in validation_plots/ directory:\n")
# cat("  - 02_latent_correlations.png\n")
# cat("  - 04_Y_metabolite_intensities.png\n")
# cat("  - 05_viral_host_ranges.png\n")
# cat("  - 06_three_interaction_matrices.png\n")
# cat("  - summary_statistics.csv\n")


################################################################################
# NATURE-STYLE VALIDATION PLOTS WITH PANEL LABELS
################################################################################
cat("\n=== GENERATING NATURE-STYLE VALIDATION PLOTS ===\n")

# Load required libraries
library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(gridExtra)  # For combining plots

# Create output directory
dir.create("validation_plots", showWarnings = FALSE)

# Nature style theme - clean, minimal, publication-ready
theme_nature <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(color = "black"),
      plot.title = element_text(size = base_size + 4, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size, hjust = 0, color = "gray30"),
      axis.title = element_text(size = base_size, face = "plain"),
      axis.text = element_text(size = base_size - 2, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      legend.title = element_text(size = base_size, face = "plain"),
      legend.text = element_text(size = base_size - 2),
      legend.key.size = unit(0.5, "cm"),
      legend.background = element_blank(),
      plot.margin = margin(10, 10, 10, 10),
      strip.text = element_text(size = base_size, face = "bold")
    )
}

################################################################################
# FIGURE 1: THREE INTERACTION MATRICES (a, b, c)
################################################################################
cat("  Creating Figure 1: Three interaction matrices (Nature style)...\n")

# Panel a: Cij (Bacteria-Metabolite interactions)
cij_df <- expand.grid(
  bacterium = 1:N_bac,
  metabolite = 1:N_met
)
cij_df$value <- as.vector(Cij)
cij_active <- cij_df %>% filter(value > 0)

p1a <- ggplot(cij_active, aes(x = bacterium, y = metabolite)) +
  geom_tile(aes(fill = value), color = NA) +
  scale_fill_viridis_c(option = "plasma", name = "Consumption\nrate") +
  scale_x_continuous(breaks = seq(0, N_bac, by = 50), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, N_met, by = 10), expand = c(0, 0)) +
  labs(
    title = "a",
    subtitle = sprintf("Bacteria-Metabolite (%.1f%% density)", 100 * nrow(cij_active) / (N_bac * N_met)),
    x = "Bacterial species",
    y = "Metabolite"
  ) +
  theme_nature() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_blank()
  )

# Panel b: C_virus (Viral AMG-Metabolite interactions)
cvirus_df <- expand.grid(
  virus = 1:N_vir,
  metabolite = 1:N_met
)
cvirus_df$value <- as.vector(C_virus)
cvirus_active <- cvirus_df %>% filter(value > 0)

p1b <- ggplot(cvirus_active, aes(x = virus, y = metabolite)) +
  geom_tile(aes(fill = value), color = NA) +
  scale_fill_viridis_c(option = "viridis", name = "Interaction\nstrength") +
  scale_x_continuous(breaks = seq(0, N_vir, by = 50), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, N_met, by = 10), expand = c(0, 0)) +
  labs(
    title = "b",
    subtitle = sprintf("Virus-Metabolite (%.1f%% density)", 100 * nrow(cvirus_active) / (N_vir * N_met)),
    x = "Viral phage",
    y = "Metabolite"
  ) +
  theme_nature() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_blank()
  )

# Panel c: Infection (Virus-Bacteria network)
infection_df <- expand.grid(
  virus = 1:N_vir,
  bacterium = 1:N_bac
)
infection_df$infects <- as.vector(t(Infection))
infection_active <- infection_df %>% filter(infects == 1)

p1c <- ggplot(infection_active, aes(x = bacterium, y = virus)) +
  geom_tile(fill = "#3C78B4", alpha = 0.8, color = NA) +
  scale_x_continuous(breaks = seq(0, N_bac, by = 50), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, N_vir, by = 50), expand = c(0, 0)) +
  labs(
    title = "c",
    subtitle = sprintf("Infection Matrix (%.1f%% density)", 100 * sum(Infection) / (N_vir * N_bac)),
    x = "Bacterial host",
    y = "Viral phage"
  ) +
  theme_nature() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_blank()
  )

# Combine with gridExtra
fig1_combined <- grid.arrange(p1a, p1b, p1c, ncol = 3)

# Save Figure 1
ggsave("validation_plots/fig1_interaction_matrices_nature.png", fig1_combined,
       width = 14, height = 5, dpi = 300)
ggsave("validation_plots/fig1_interaction_matrices_nature.pdf", fig1_combined,
       width = 14, height = 5)
cat("    Saved to validation_plots/fig1_interaction_matrices_nature.png/pdf\n")


################################################################################
# FIGURE 2: LATENT FACTOR CORRELATION HEATMAP (single panel)
################################################################################
cat("  Creating Figure 2: Latent correlation heatmap (Nature style)...\n")

# Combine all latents into one matrix
all_latents <- cbind(Z1_store, Z2_store, Zs_store)
colnames(all_latents) <- c(
  paste0("Z1_", 1:dim_Z1_s),
  paste0("Z2_", 1:dim_Z2_s),
  paste0("Zc_", 1:dim_Zc)
)

# Compute correlation matrix
cor_matrix <- cor(all_latents)

# Convert to long format for ggplot
cor_df <- as.data.frame(as.table(cor_matrix))
colnames(cor_df) <- c("Var1", "Var2", "value")

# Compute mean correlations for subtitle
z1_indices <- 1:dim_Z1_s
z2_indices <- (dim_Z1_s + 1):(dim_Z1_s + dim_Z2_s)
zc_indices <- (dim_Z1_s + dim_Z2_s + 1):(dim_Z1_s + dim_Z2_s + dim_Zc)

mean_z1_z2 <- mean(abs(cor_matrix[z1_indices, z2_indices]))
mean_z1_zc <- mean(abs(cor_matrix[z1_indices, zc_indices]))
mean_z2_zc <- mean(abs(cor_matrix[z2_indices, zc_indices]))

# Create heatmap
fig2 <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "#2166ac",
    mid = "white",
    high = "#b2182b",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Correlation"
  ) +
  # Add dashed lines to separate Z1/Z2/Zc blocks
  geom_vline(xintercept = dim_Z1_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_vline(xintercept = dim_Z1_s + dim_Z2_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_hline(yintercept = dim_Z1_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_hline(yintercept = dim_Z1_s + dim_Z2_s + 0.5, linetype = "dashed", color = "black", linewidth = 0.5) +
  # Add block labels (using Unicode subscripts for compatibility)
  annotate("text", x = dim_Z1_s/2, y = -1.5, label = "Z\u2081", size = 3, fontface = "bold") +
  annotate("text", x = dim_Z1_s + dim_Z2_s/2, y = -1.5, label = "Z\u2082", size = 3, fontface = "bold") +
  annotate("text", x = dim_Z1_s + dim_Z2_s + dim_Zc/2, y = -1.5, label = "Zc", size = 3, fontface = "bold") +
  labs(
    subtitle = sprintf("Mean |r|: Z1-Z2=%.3f, Z1-Zc=%.3f, Z2-Zc=%.3f",
                       mean_z1_z2, mean_z1_zc, mean_z2_zc),
    x = "Latent factors",
    y = "Latent factors"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
    axis.text.y = element_text(size = 6),
    legend.position = "right",
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_blank()
  ) +
  coord_fixed()

# Save Figure 2
ggsave("validation_plots/fig2_latent_correlations_nature.png", fig2,
       width = 120, height = 100, units = "mm", dpi = 300)
ggsave("validation_plots/fig2_latent_correlations_nature.pdf", fig2,
       width = 120, height = 100, units = "mm")
cat("    Saved to validation_plots/fig2_latent_correlations_nature.png/pdf\n")


################################################################################
# FIGURE 3: METABOLITE DISTRIBUTIONS + VIRAL HOST RANGE (a, b)
################################################################################
cat("  Creating Figure 3: Metabolite distributions and viral host range (Nature style)...\n")

# Read ground truth to get variance shares for coloring
gt_temp <- read.csv("GT_virome_variance_shares_complex_COPSAC.csv")

# Classify metabolites by dominant variance source
dom_thr <- 0.35
met_groups <- rep("Mixed", N_met)
met_groups[gt_temp$share_Z1 >= dom_thr &
           gt_temp$share_Z1 >= gt_temp$share_Z2 &
           gt_temp$share_Z1 >= gt_temp$share_Zs &
           gt_temp$share_Z1 >= gt_temp$share_noise] <- "Bacteria"

met_groups[gt_temp$share_Z2 >= dom_thr &
           gt_temp$share_Z2 >= gt_temp$share_Z1 &
           gt_temp$share_Z2 >= gt_temp$share_Zs &
           gt_temp$share_Z2 >= gt_temp$share_noise] <- "Virus"

met_groups[gt_temp$share_Zs >= dom_thr &
           gt_temp$share_Zs >= gt_temp$share_Z1 &
           gt_temp$share_Zs >= gt_temp$share_Z2 &
           gt_temp$share_Zs >= gt_temp$share_noise] <- "Shared"

met_groups[gt_temp$share_noise >= dom_thr &
           gt_temp$share_noise >= gt_temp$share_Z1 &
           gt_temp$share_noise >= gt_temp$share_Z2 &
           gt_temp$share_noise >= gt_temp$share_Zs] <- "Noise"

# Create long-format data for plotting
Y_long <- Y_df %>%
  select(-sim, -rep) %>%
  pivot_longer(cols = everything(), names_to = "metabolite", values_to = "abundance") %>%
  mutate(
    met_num = as.numeric(gsub("met_", "", metabolite)),
    group = factor(met_groups[met_num], levels = c("Bacteria", "Virus", "Shared", "Noise", "Mixed"))
  )

# Panel a: Metabolite intensity distributions
p3a <- ggplot(Y_long, aes(x = reorder(metabolite, met_num), y = abundance, fill = group)) +
  geom_boxplot(alpha = 0.85, outlier.size = 0.2, outlier.alpha = 0.3, linewidth = 0.2) +
  scale_fill_manual(
    values = c(
      "Bacteria" = "#3C78B4",
      "Virus" = "#C44E52",
      "Shared" = "#8172B3",
      "Noise" = "#8C8C8C",
      "Mixed" = "#CCB974"
    ),
    name = "Dominant\nsource"
  ) +
  labs(
    title = "a",
    subtitle = sprintf("Metabolite intensity distributions (n = %d samples, %d metabolites)", n_sims, N_met),
    x = "Metabolite",
    y = "Abundance"
  ) +
  theme_nature() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right"
  )

# Panel b: Viral host range distribution
hosts_per_virus <- rowSums(Infection)
virus_selectivity_class <- ifelse(hosts_per_virus == 1, "Specialist",
                            ifelse(hosts_per_virus <= 10, "Moderate", "Generalist"))

virus_host_df <- data.frame(
  virus = 1:N_vir,
  n_hosts = hosts_per_virus,
  selectivity = factor(virus_selectivity_class, levels = c("Specialist", "Moderate", "Generalist"))
)

selectivity_counts <- table(virus_selectivity_class)

p3b <- ggplot(virus_host_df, aes(x = n_hosts, fill = selectivity)) +
  geom_histogram(binwidth = 2, alpha = 0.85, color = "white", linewidth = 0.2) +
  scale_fill_manual(
    values = c(
      "Specialist" = "#C44E52",
      "Moderate" = "#DD8452",
      "Generalist" = "#55A868"
    ),
    name = "Host range"
  ) +
  labs(
    title = "b",
    subtitle = sprintf("Viral host range (Specialists: %d, Moderate: %d, Generalists: %d)",
                      selectivity_counts["Specialist"],
                      selectivity_counts["Moderate"],
                      selectivity_counts["Generalist"]),
    x = "Number of bacterial hosts",
    y = "Count"
  ) +
  theme_nature() +
  theme(legend.position = "right")

# Combine with gridExtra (vertical stacking)
fig3_combined <- grid.arrange(p3a, p3b, ncol = 1, heights = c(1, 0.8))

# Save Figure 3
ggsave("validation_plots/fig3_metabolites_viral_hosts_nature.png", fig3_combined,
       width = 12, height = 10, dpi = 300)
ggsave("validation_plots/fig3_metabolites_viral_hosts_nature.pdf", fig3_combined,
       width = 12, height = 10)
cat("    Saved to validation_plots/fig3_metabolites_viral_hosts_nature.png/pdf\n")


################################################################################
# SUMMARY STATISTICS CSV
################################################################################
cat("  Creating summary statistics CSV...\n")

total_infections <- sum(Infection)
infection_density <- 100 * total_infections / (N_vir * N_bac)
hosts_per_virus <- rowSums(Infection)
mean_hosts <- mean(hosts_per_virus)
sd_hosts <- sd(hosts_per_virus)

summary_stats <- data.frame(
  Metric = c(
    "Number of simulations",
    "Number of bacteria",
    "Number of viruses",
    "Number of metabolites",
    "Z1 dimension (bacteria-specific)",
    "Z2 dimension (virus-specific)",
    "Zc dimension (shared)",
    "Total infections",
    "Infection density (%)",
    "Mean hosts per virus",
    "SD hosts per virus",
    "Total AMGs",
    "Mean |cor(Z1,Z2)|",
    "Mean |cor(Z1,Zc)|",
    "Mean |cor(Z2,Zc)|"
  ),
  Value = c(
    n_sims,
    N_bac,
    N_vir,
    N_met,
    dim_Z1_s,
    dim_Z2_s,
    dim_Zc,
    total_infections,
    round(infection_density, 2),
    round(mean_hosts, 2),
    round(sd_hosts, 2),
    sum(C_virus > 0),
    round(mean_z1_z2, 4),
    round(mean_z1_zc, 4),
    round(mean_z2_zc, 4)
  )
)

write.csv(summary_stats, "validation_plots/summary_statistics.csv", row.names = FALSE)
cat("    Saved to validation_plots/summary_statistics.csv\n")

cat("\n=== ALL NATURE-STYLE VALIDATION PLOTS COMPLETE ===\n")
cat("Plots saved in validation_plots/ directory:\n")
cat("  - fig1_interaction_matrices_nature.png/pdf (panels a, b, c)\n")
cat("  - fig2_latent_correlations_nature.png/pdf\n")
cat("  - fig3_metabolites_viral_hosts_nature.png/pdf (panels a, b)\n")
cat("  - summary_statistics.csv\n")
