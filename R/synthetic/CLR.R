cat("\n=== CLR TRANSFORMATION WITH ALIGNMENT ON (sim, rep) ===\n")

X1_df0 <- read.csv(
  "X1_bacteria_synthetic_RA_complex_COPSAC.csv",
  sep = ",", header = TRUE, check.names = FALSE
)

X2_df0 <- read.csv(
  "X2_viruses_synthetic_RA_complex_COPSAC.csv",
  sep = ",", header = TRUE, check.names = FALSE
)

library(compositions)
library(zCompositions)
library(caret)

# ------------------------------------------------------------
# 1) ALIGN ON (sim, rep)
# ------------------------------------------------------------
if (!all(c("sim", "rep") %in% colnames(X1_df0))) stop("X1 is missing sim/rep columns.")
if (!all(c("sim", "rep") %in% colnames(X2_df0))) stop("X2 is missing sim/rep columns.")

X1_df0$sim <- as.integer(X1_df0$sim)
X1_df0$rep <- as.integer(X1_df0$rep)
X2_df0$sim <- as.integer(X2_df0$sim)
X2_df0$rep <- as.integer(X2_df0$rep)

X1_df0$id <- paste(X1_df0$sim, X1_df0$rep, sep = "__")
X2_df0$id <- paste(X2_df0$sim, X2_df0$rep, sep = "__")

# Sanity: duplicated keys will break alignment
if (any(duplicated(X1_df0$id))) stop("X1 has duplicated (sim,rep) keys.")
if (any(duplicated(X2_df0$id))) stop("X2 has duplicated (sim,rep) keys.")

common_ids <- intersect(X1_df0$id, X2_df0$id)

cat(sprintf("X1 rows: %d\n", nrow(X1_df0)))
cat(sprintf("X2 rows: %d\n", nrow(X2_df0)))
cat(sprintf("Common (sim,rep) pairs: %d\n", length(common_ids)))

if (length(common_ids) == 0) stop("No overlapping (sim,rep) pairs found between X1 and X2.")

# Subset both to common ids, then order identically
X1_aligned <- X1_df0[X1_df0$id %in% common_ids, , drop = FALSE]
X2_aligned <- X2_df0[X2_df0$id %in% common_ids, , drop = FALSE]

# Order by id (or by sim then rep, either is fine as long as same)
ord <- order(common_ids)
common_ids_sorted <- common_ids[ord]

X1_aligned <- X1_aligned[match(common_ids_sorted, X1_aligned$id), , drop = FALSE]
X2_aligned <- X2_aligned[match(common_ids_sorted, X2_aligned$id), , drop = FALSE]

# Final sanity check: ids match row-by-row
if (!identical(X1_aligned$id, X2_aligned$id)) stop("Alignment failed: X1 and X2 ids are not identical after ordering.")

# Keep sim/rep vectors (now aligned)
sim_col <- X1_aligned$sim
rep_col <- X1_aligned$rep

# ------------------------------------------------------------
# 2) SPLIT TAXA (exclude sim/rep/id)
# ------------------------------------------------------------
X1_taxa <- as.matrix(X1_aligned[, !(colnames(X1_aligned) %in% c("sim", "rep", "id")), drop = FALSE])
X2_taxa <- as.matrix(X2_aligned[, !(colnames(X2_aligned) %in% c("sim", "rep", "id")), drop = FALSE])


# ------------------------------------------------------------
# 3) FILTER + MULTIPLICATIVE REPLACEMENT + CLR
# ------------------------------------------------------------



# STEP 1: NEAR-ZERO VARIANCE FILTERING ON TAXA (columns), NOT SAMPLES (rows)
clr_transform <- function(x, min_prevalence = 0.10) {
  x <- as.matrix(x)
  n_taxa_original <- ncol(x)

  # 1) Near-zero variance taxa filtering (columns)
  nzv_taxa <- caret::nearZeroVar(x, names = FALSE)
  if (length(nzv_taxa) > 0) {
    cat(sprintf("  Near-zero variance filtering removed %d taxa\n", length(nzv_taxa)))
    x <- x[, -nzv_taxa, drop = FALSE]
  } else {
    cat("  No near-zero variance taxa found\n")
  }

  # 2) Prevalence filtering (columns)
  prevalence <- colMeans(x > 0)
  taxa_to_keep <- prevalence >= min_prevalence

  cat(sprintf("  Total taxa (before filters): %d\n", n_taxa_original))
  cat(sprintf("  Taxa present in >= %.0f%% of samples: %d\n", min_prevalence * 100, sum(taxa_to_keep)))
  cat(sprintf("  Taxa removed (low prevalence): %d\n", sum(!taxa_to_keep)))

  x <- x[, taxa_to_keep, drop = FALSE]
  if (ncol(x) == 0) stop("No taxa remain after prevalence filtering. Lower min_prevalence.")

  # 3) Renormalize rows
  row_sums <- rowSums(x)
  if (any(row_sums == 0)) stop("Some samples have zero total abundance after filtering.")
  x <- x / row_sums

  # 4) Multiplicative replacement for zeros
  if (any(x == 0)) {
    cat(sprintf("  Found %d zeros, applying multiplicative replacement...\n", sum(x == 0)))
    x <- zCompositions::cmultRepl(
      x,
      label = 0,
      method = "CZM",
      z.delete = FALSE,
      z.warning = 0
    )
  } else {
    cat("  No zeros found, skipping multiplicative replacement\n")
  }

  # 5) CLR
  as.matrix(compositions::clr(compositions::acomp(x)))
}

cat("\n=== FILTERING AND TRANSFORMING BACTERIA (X1) ===\n")
X1_clr_final <- clr_transform(X1_taxa, min_prevalence = 0.10)

cat("\n=== FILTERING AND TRANSFORMING VIRUSES (X2) ===\n")
X2_clr_final <- clr_transform(X2_taxa, min_prevalence = 0.10)
stopifnot(all(is.finite(X1_clr_final)))
stopifnot(all(is.finite(X2_clr_final)))

sim_final <- sim_col
rep_final <- rep_col
# ------------------------------------------------------------
# 4) WRITE OUTPUTS (aligned rows preserved)
# ------------------------------------------------------------
cat("\n=== BUILDING FINAL OUTPUT DATA FRAMES ===\n")
cat(sprintf("Final X1_clr: %d rows, %d cols\n", nrow(X1_clr_final), ncol(X1_clr_final)))
cat(sprintf("Final X2_clr: %d rows, %d cols\n", nrow(X2_clr_final), ncol(X2_clr_final)))

X1_out <- data.frame(sim = sim_final, rep = rep_final, X1_clr_final, check.names = FALSE)
X2_out <- data.frame(sim = sim_final, rep = rep_final, X2_clr_final, check.names = FALSE)

write.csv(X1_out, file = "X1_bacteria_synthetic_CLR.csv", row.names = FALSE)
write.csv(X2_out, file = "X2_viruses_synthetic_CLR.csv", row.names = FALSE)


# library(compositions)

# # ------------------------------------------------------------
# # 1) ALIGN ON (sim, rep)
# # ------------------------------------------------------------
# if (!all(c("sim", "rep") %in% colnames(X1_df0))) stop("X1 is missing sim/rep columns.")
# if (!all(c("sim", "rep") %in% colnames(X2_df0))) stop("X2 is missing sim/rep columns.")

# X1_df0$sim <- as.integer(X1_df0$sim)
# X1_df0$rep <- as.integer(X1_df0$rep)
# X2_df0$sim <- as.integer(X2_df0$sim)
# X2_df0$rep <- as.integer(X2_df0$rep)

# X1_df0$id <- paste(X1_df0$sim, X1_df0$rep, sep = "__")
# X2_df0$id <- paste(X2_df0$sim, X2_df0$rep, sep = "__")

# # Sanity: duplicated keys will break alignment
# if (any(duplicated(X1_df0$id))) stop("X1 has duplicated (sim,rep) keys.")
# if (any(duplicated(X2_df0$id))) stop("X2 has duplicated (sim,rep) keys.")

# common_ids <- intersect(X1_df0$id, X2_df0$id)

# cat(sprintf("X1 rows: %d\n", nrow(X1_df0)))
# cat(sprintf("X2 rows: %d\n", nrow(X2_df0)))
# cat(sprintf("Common (sim,rep) pairs: %d\n", length(common_ids)))

# if (length(common_ids) == 0) stop("No overlapping (sim,rep) pairs found between X1 and X2.")

# # Subset both to common ids, then order identically
# X1_aligned <- X1_df0[X1_df0$id %in% common_ids, , drop = FALSE]
# X2_aligned <- X2_df0[X2_df0$id %in% common_ids, , drop = FALSE]

# # Order by id (or by sim then rep, either is fine as long as same)
# ord <- order(common_ids)
# common_ids_sorted <- common_ids[ord]

# X1_aligned <- X1_aligned[match(common_ids_sorted, X1_aligned$id), , drop = FALSE]
# X2_aligned <- X2_aligned[match(common_ids_sorted, X2_aligned$id), , drop = FALSE]

# # Final sanity check: ids match row-by-row
# if (!identical(X1_aligned$id, X2_aligned$id)) stop("Alignment failed: X1 and X2 ids are not identical after ordering.")

# # Keep sim/rep vectors (now aligned)
# sim_col <- X1_aligned$sim
# rep_col <- X1_aligned$rep

# # ------------------------------------------------------------
# # 2) SPLIT TAXA (exclude sim/rep/id)
# # ------------------------------------------------------------
# X1_taxa <- as.matrix(X1_aligned[, !(colnames(X1_aligned) %in% c("sim", "rep", "id")), drop = FALSE])
# X2_taxa <- as.matrix(X2_aligned[, !(colnames(X2_aligned) %in% c("sim", "rep", "id")), drop = FALSE])

# # ------------------------------------------------------------
# # 3) CLR TRANSFORMATION WITH PSEUDOCOUNT 1e-6
# # ------------------------------------------------------------
# clr_transform <- function(x, pseudocount = 1e-6) {
#   x <- as.matrix(x)

#   # Add pseudocount to handle zeros
#   x_pseudo <- x + pseudocount

#   # Apply CLR transformation
#   clr_result <- log(x_pseudo) - rowMeans(log(x_pseudo))

#   return(clr_result)
# }

# ------------------------------------------------------------
# 4) APPLY CLR ON ALIGNED DATA
# ------------------------------------------------------------
# cat("\n=== TRANSFORMING BACTERIA (X1) WITH CLR (pseudocount = 1e-6) ===\n")
# X1_clr_final <- clr_transform(X1_taxa, pseudocount = 1e-6)

# cat("\n=== TRANSFORMING VIRUSES (X2) WITH CLR (pseudocount = 1e-6) ===\n")
# X2_clr_final <- clr_transform(X2_taxa, pseudocount = 1e-6)

# sim_final <- sim_col
# rep_final <- rep_col




# cat("\n=== BUILDING FINAL OUTPUT DATA FRAMES ===\n")
# cat(sprintf("Final X1_clr: %d rows, %d cols\n", nrow(X1_clr_final), ncol(X1_clr_final)))
# cat(sprintf("Final X2_clr: %d rows, %d cols\n", nrow(X2_clr_final), ncol(X2_clr_final)))

# X1_out <- data.frame(sim = sim_final, rep = rep_final, X1_clr_final, check.names = FALSE)
# X2_out <- data.frame(sim = sim_final, rep = rep_final, X2_clr_final, check.names = FALSE)

# write.csv(X1_out, file = "X1_bacteria_synthetic_CLR_sparse.csv", row.names = FALSE)
# write.csv(X2_out, file = "X2_viruses_synthetic_CLR_sparse.csv", row.names = FALSE)
