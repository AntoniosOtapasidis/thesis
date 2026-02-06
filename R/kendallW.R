# ============================================================
# Kendall's W across runs (R) for your LEAF variance CSVs
# RUN_DIR:  /users/antonios/LEAF_revisit/LEAF/one_hold/2_layers/
# PATTERN:  5000_variance_comparison_*.csv
# FRACTION_COL: test
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(irr)      # kendall()
})

RUN_DIR <- "/users/antonios/LEAF_revisit/LEAF/one_hold/2_layers/256-64-5000-32-50/"
PATTERN <- "5000_Variance_comparison_.*\\.csv$"
FRACTION_COL <- "test"

COMPONENT_RENAME <- c(
  "z1" = "z1",
  "zs1" = "z1",
  "zc1" = "zc",
  "zc" = "zc",
  "zs2" = "zs2",
  "z2" = "zs2",
  "noise" = "noise"
)

COMPONENT_ORDER <- c("z1", "zc", "zs2")

infer_run_id <- function(path) {
  fname <- basename(path)
  m <- str_match(fname, "seed(\\d+)")
  if (!is.na(m[1, 2])) return(m[1, 2])
  return(fname)
}

# ---------- Load all runs ----------
paths <- list.files(RUN_DIR, pattern = PATTERN, full.names = TRUE)
if (length(paths) == 0) stop("No files found. Check RUN_DIR or PATTERN.")

all_df <- map_dfr(paths, function(p) {
  read_csv(p, show_col_types = FALSE) %>%
    mutate(run_id = infer_run_id(p))
})

required_cols <- c("outcome", "component", FRACTION_COL, "run_id")
missing_cols <- setdiff(required_cols, colnames(all_df))
if (length(missing_cols) > 0) {
  stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
}

all_df <- all_df %>%
  mutate(
    component_clean = tolower(as.character(component)),
    component_clean = if_else(component_clean %in% names(COMPONENT_RENAME),
                              COMPONENT_RENAME[component_clean],
                              component_clean),
    value = suppressWarnings(as.numeric(.data[[FRACTION_COL]]))
  ) %>%
  filter(!is.na(value)) %>%
  filter(component_clean %in% COMPONENT_ORDER)

runs <- sort(unique(all_df$run_id))
cat("Runs detected:", length(runs), "\n")
outcomes <- sort(unique(all_df$outcome))
cat("Metabolites detected:", length(outcomes), "\n")

# ---------- Kendall W per outcome ----------
# Note: irr::kendall expects a matrix with rows=objects (items), cols=judges (raters)
# Here: objects = components, judges = runs
compute_kendall_w <- function(df_one_outcome) {

  wide <- df_one_outcome %>%
    group_by(run_id, component_clean) %>%
    summarise(value = sum(value), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = component_clean, values_from = value, values_fill = 0) %>%
    select(run_id, all_of(COMPONENT_ORDER))

  # Ensure deterministic run ordering
  wide <- wide %>% arrange(run_id)

  # Convert to rank matrix per run: rank 1 = largest value
  vals_mat <- as.matrix(wide %>% select(all_of(COMPONENT_ORDER)))
  ranks_mat <- t(apply(vals_mat, 1, function(x) rank(-x, ties.method = "average")))

  # irr::kendall wants rows=components, cols=runs, so transpose
  W_res <- irr::kendall(t(ranks_mat))

  tibble(
    metabolite = unique(df_one_outcome$outcome),
    kendall_W = as.numeric(W_res$value),
    p_value = as.numeric(W_res$p.value),
    n_runs = nrow(ranks_mat)
  )
}

results_w <- all_df %>%
  group_by(outcome) %>%
  group_modify(~ compute_kendall_w(.x)) %>%
  ungroup()

print(results_w)

# Optional: save
out_path <- file.path(RUN_DIR, "Final_kendallW_irr_results.csv")
write_csv(results_w, out_path)
cat("Saved:", out_path, "\n")
