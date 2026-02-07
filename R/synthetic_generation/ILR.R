# 1) Build balanced SBP for a given set of features
sbp_balanced <- function(features) {
  D <- length(features)
  if (D < 2) stop(sprintf("Need at least 2 features for SBP, got D = %d", D))

  splits <- list()
  q <- list(seq_len(D))  # queue of index vectors

  while (length(q) > 0) {
    grp <- q[[1]]; q <- q[-1]
    if (length(grp) <= 1) next
    k <- length(grp)
    k_left <- floor(k / 2)
    left  <- grp[seq_len(k_left)]
    right <- grp[(k_left + 1):k]
    splits[[length(splits) + 1]] <- list(num = left, den = right)
    q[[length(q) + 1]] <- left
    q[[length(q) + 1]] <- right
  }

  if (length(splits) != D - 1) {
    stop(sprintf("Balanced SBP generated %d splits, expected %d.", length(splits), D - 1))
  }

  sbp <- matrix(0L, nrow = D, ncol = D - 1,
                dimnames = list(features, paste0("split", seq_len(D - 1))))
  for (j in seq_along(splits)) {
    sbp[splits[[j]]$num, j] <-  1L
    sbp[splits[[j]]$den, j] <- -1L
  }
  sbp
}


# 2) Convert SBP to orthonormal ILR basis
sbp_to_basis <- function(sbp) {
  sbp <- as.matrix(sbp)
  D <- nrow(sbp); K <- ncol(sbp)
  if (K != D - 1) stop("SBP must have D rows and D-1 columns.")
  B <- matrix(0, nrow = D, ncol = K)
  for (j in seq_len(K)) {
    r <- sum(sbp[, j] ==  1L)
    s <- sum(sbp[, j] == -1L)
    if (r == 0 || s == 0) stop(sprintf("Split %d must have both +1 and -1 groups.", j))
    B[sbp[, j] ==  1L, j] <-  sqrt(s / (r * (r + s)))
    B[sbp[, j] == -1L, j] <- -sqrt(r / (s * (r + s)))
  }
  B
}

# 3) ILR using that basis
ilr_transform <- function(M, sbp, min_prevalence = 0.10) {
  M <- as.matrix(M)  # samples in rows, features in columns

  # =================================================================
  # PREVALENCE FILTERING: Keep taxa present in at least min_prevalence of samples
  # =================================================================
  n_samples <- nrow(M)
  prevalence <- colSums(M > 0) / n_samples

  taxa_to_keep <- prevalence >= min_prevalence

  cat(sprintf("  Total taxa: %d\n", ncol(M)))
  cat(sprintf("  Taxa present in >= %.0f%% of samples: %d\n",
              min_prevalence * 100, sum(taxa_to_keep)))
  cat(sprintf("  Taxa removed (low prevalence): %d\n", sum(!taxa_to_keep)))

  # Filter taxa and corresponding SBP rows
  M_filtered <- M[, taxa_to_keep, drop = FALSE]
  sbp_filtered <- sbp[taxa_to_keep, , drop = FALSE]

  # Remove SBP columns that have all zeros (splits involving only removed taxa)
  col_has_data <- colSums(abs(sbp_filtered)) > 0
  sbp_filtered <- sbp_filtered[, col_has_data, drop = FALSE]

  D <- ncol(M_filtered)
  if (D < 2) stop("Need at least 2 features for ILR after filtering.")

  # Rebuild balanced SBP for remaining taxa (to ensure proper structure)
  sbp_new <- sbp_balanced(colnames(M_filtered))

  cat(sprintf("  Rebuilding balanced SBP for %d remaining taxa\n", D))
  cat(sprintf("  New SBP dimensions: %d taxa x %d splits\n", nrow(sbp_new), ncol(sbp_new)))

  # Renormalize after filtering (compositions must sum to 1)
  row_sums <- rowSums(M_filtered)
  if (any(row_sums == 0)) {
    warning("Some samples have zero total abundance after filtering!")
  }
  M_filtered <- M_filtered / row_sums

  # =================================================================
  # MULTIPLICATIVE REPLACEMENT: Handle zeros (DO NOT DELETE ROWS)
  # =================================================================
  if (any(M_filtered == 0)) {
    cat("  Applying multiplicative replacement for zeros...\n")
    M_replaced <- zCompositions::cmultRepl(
      M_filtered,
      label = 0,
      method = "CZM",
      z.delete = FALSE,
      z.warning = 0
    )
    M_replaced <- as.matrix(M_replaced)
  } else {
    cat("  No zeros detected, skipping multiplicative replacement.\n")
    M_replaced <- M_filtered
  }

  # =================================================================
  # ILR TRANSFORMATION
  # =================================================================
  B <- sbp_to_basis(sbp_new)
  rownames(B) <- colnames(M_replaced)

  L   <- log(M_replaced)
  clr <- sweep(L, 1, rowMeans(L), "-")  # row-wise centering
  clr <- as.matrix(clr)

  Z <- clr %*% B
  rownames(Z) <- rownames(M_replaced)
  colnames(Z) <- colnames(B)
  Z
}

################################################################################
# APPLY BALANCED ILR TO SYNTHETIC X1_mat, X2_mat (ALIGNED ON sim/rep)
################################################################################

library(zCompositions)

cat("\n=== ILR TRANSFORMATION WITH BALANCED SBP ON SYNTHETIC DATA ===\n")
X1_mat <- read.csv(
  "X1_bacteria_synthetic_RA_complex_COPSAC.csv",
  sep = ",", header = TRUE, check.names = FALSE
)

X2_mat <- read.csv(
  "X2_viruses_synthetic_RA_complex_COPSAC.csv",
  sep = ",", header = TRUE, check.names = FALSE
)

# -----------------------------
# ALIGN ON (sim, rep)
# -----------------------------
if (!all(c("sim", "rep") %in% colnames(X1_mat))) stop("X1 is missing sim/rep columns.")
if (!all(c("sim", "rep") %in% colnames(X2_mat))) stop("X2 is missing sim/rep columns.")

X1_mat$sim <- as.integer(X1_mat$sim)
X1_mat$rep <- as.integer(X1_mat$rep)
X2_mat$sim <- as.integer(X2_mat$sim)
X2_mat$rep <- as.integer(X2_mat$rep)

X1_mat$id <- paste(X1_mat$sim, X1_mat$rep, sep = "__")
X2_mat$id <- paste(X2_mat$sim, X2_mat$rep, sep = "__")

if (any(duplicated(X1_mat$id))) stop("X1 has duplicated (sim,rep) keys.")
if (any(duplicated(X2_mat$id))) stop("X2 has duplicated (sim,rep) keys.")

common_ids <- intersect(X1_mat$id, X2_mat$id)
if (length(common_ids) == 0) stop("No overlapping (sim,rep) pairs found between X1 and X2.")

common_ids_sorted <- sort(common_ids)

X1_aligned <- X1_mat[match(common_ids_sorted, X1_mat$id), , drop = FALSE]
X2_aligned <- X2_mat[match(common_ids_sorted, X2_mat$id), , drop = FALSE]

if (!identical(X1_aligned$id, X2_aligned$id)) stop("Alignment failed: ids not identical after ordering.")

# split metadata and taxa (now aligned)
sim_col <- X1_aligned$sim
rep_col <- X1_aligned$rep

X1_taxa <- as.matrix(X1_aligned[, !(names(X1_aligned) %in% c("sim", "rep", "id")), drop = FALSE])
X2_taxa <- as.matrix(X2_aligned[, !(names(X2_aligned) %in% c("sim", "rep", "id")), drop = FALSE])

# build balanced SBP on TAXA ONLY (will be rebuilt inside ilr_transform after filtering)
sbp_bac <- sbp_balanced(colnames(X1_taxa))
sbp_vir <- sbp_balanced(colnames(X2_taxa))

# ILR transform with prevalence filtering and multiplicative replacement
cat("\n=== FILTERING AND TRANSFORMING BACTERIA (X1) ===\n")
X1_ilr <- ilr_transform(X1_taxa, sbp = sbp_bac, min_prevalence = 0.10)

cat("\n=== FILTERING AND TRANSFORMING VIRUSES (X2) ===\n")
X2_ilr <- ilr_transform(X2_taxa, sbp = sbp_vir, min_prevalence = 0.10)

cat("=== ILR TRANSFORMATION COMPLETE ===\n")
cat(sprintf("X1_ilr dimensions: %d samples x %d ILR coords\n", nrow(X1_ilr), ncol(X1_ilr)))
cat(sprintf("X2_ilr dimensions: %d samples x %d ILR coords\n", nrow(X2_ilr), ncol(X2_ilr)))

cat("\n=== BUILDING FINAL OUTPUT DATA FRAMES ===\n")

# rebuild data frames with sim/rep + ILR coords (aligned keys)
X1_df <- data.frame(sim = sim_col,
                    rep = rep_col,
                    X1_ilr,
                    check.names = FALSE)

X2_df <- data.frame(sim = sim_col,
                    rep = rep_col,
                    X2_ilr,
                    check.names = FALSE)

write.csv(X1_df, file = "X1_bacteria_synthetic_ILR_final.csv", row.names = FALSE)
write.csv(X2_df, file = "X2_viruses_synthetic_ILR_final.csv",  row.names = FALSE)
