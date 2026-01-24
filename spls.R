---
title: "Virome"
format: html
editor: visual
---

## Virome sPLS

```{r}
library(ggplot2)
library(tidyverse)
library(tidyr)
## install BiocManager if not installed 
if (!requireNamespace("BiocManager", quietly = TRUE))    
#install.packages("BiocManager")
## then install mixOmics
#BiocManager::install("mixOmics")
library(mixOmics)
library(vegan)
```

```{r}
#virome data after multiplicative replacement and CLR transformation
virome <-read.table("~/data/virome_families_for_multi_repl.tsv",check.names = FALSE)
virome
```

```{r}
metabolites <- read.table("~/data/metabolites.known.tsv", sep = "\t", header = TRUE)
metabolites
```

Pair the datasets

```{r}
virome_T <- as.data.frame(t(virome))
# Suppose your dataframe is called df
# base R
#colnames(bacteriome_T) <- bacteriome_T[1, ]   # take first row as column names
#bacteriome_T <- bacteriome_T[-1, ]            # remove the first row
               # Remove that first row


virome_T
```

```{r}
id_col <-"id"
# 3) Move IDs to rownames in metabolites and drop the ID column
metabolites2 <- as.data.frame(metabolites, check.names = FALSE)
rownames(metabolites2) <- as.character(metabolites2[[id_col]])
metabolites2[[id_col]] <- NULL
```

```{r}
dim(metabolites2)
```

```{r}
virome_T <- as.data.frame(t(virome))
dim(virome_T)
dim(metabolites2)


```

```{r}
# 4) Ensure both objects are data.frames with IDs as rownames
virome_T <- as.data.frame(virome_T, check.names = FALSE)
metabolites2  <- as.data.frame(metabolites2,  check.names = FALSE)
```

```{r}
head(metabolites2)
```

```{r}
# 5) Keep only matching IDs and sort them identically
common_ids <- intersect(rownames(virome_T), rownames(metabolites2))

if (length(common_ids) == 0) {
  stop("No matching IDs between virome columns and metabolite ID column.")
}

common_ids <- sort(common_ids)

virome_aligned  <- virome_T [common_ids, , drop = FALSE]
metabolome_aligned  <- metabolites2 [common_ids, , drop = FALSE]
```

```{r}
dim(virome_aligned)
dim(metabolome_aligned)
```

```{r}
virome <- virome_aligned
metabolome <- metabolome_aligned

head(virome)
head(metabolome)
```

Prefilter on the virome data

```{r}
virome_T <- t(virome)
virome_T <-as.data.frame(virome_T)

virome_T
```

```{r}

min_present <- ceiling(0.10 * ncol(virome_T))  # 10% of individuals

present_counts <- rowSums(
  virome_T != 0 & !is.na(virome_T)
)

virome_filtered <- virome_T[present_counts >= min_present, ]

dim(virome_filtered)

virome <- virome_filtered
virome
```

```{r}
X <- data.matrix(virome)      # converts non-numerics to NA if needed
mode(X) <- "numeric"

# 1) Columns (samples)
cols_all_na    <- colSums(is.na(X)) == nrow(X)
cols_all_zero  <- !cols_all_na & colSums(replace(X, is.na(X), 0)) == 0

cat("== COLUMN CHECKS (samples) ==\n")
cat("All-NA columns (n=", sum(cols_all_na), "):\n", sep="")
if (any(cols_all_na)) print(colnames(X)[cols_all_na]) else cat("None\n")
cat("All-zero (or NA->0) columns (n=", sum(cols_all_zero), "):\n", sep="")
if (any(cols_all_zero)) print(colnames(X)[cols_all_zero]) else cat("None\n")

# 2) Rows (taxa)
rows_all_na    <- rowSums(is.na(X)) == ncol(X)
rows_all_zero  <- !rows_all_na & rowSums(replace(X, is.na(X), 0)) == 0

cat("\n== ROW CHECKS (taxa) ==\n")
cat("All-NA rows (n=", sum(rows_all_na), "):\n", sep="")
if (any(rows_all_na)) print(rownames(X)[rows_all_na]) else cat("None\n")
cat("All-zero (or NA->0) rows (n=", sum(rows_all_zero), "):\n", sep="")
if (any(rows_all_zero)) print(rownames(X)[rows_all_zero]) else cat("None\n")


```

```{r}
# Drop all-NA and all-zero rows (taxa)
rows_to_drop <- rows_all_na | rows_all_zero
cat("Dropping", sum(rows_to_drop), "rows (taxa) with all NA or all zero values.\n")

X_clean <- X[!rows_to_drop, , drop = FALSE]

# Optional: also drop problematic columns if needed
cols_to_drop <- cols_all_na | cols_all_zero
cat("Dropping", sum(cols_to_drop), "columns (samples) with all NA or all zero values.\n")

X_clean <- X_clean[, !cols_to_drop, drop = FALSE]

# Check dimensions after filtering
dim(X_clean)
```

```{r}
# Basic boxplot across all samples (columns)
boxplot(
  X_clean,
  outline = TRUE,           # show outlier points
  main = "Distribution per sample (potential outliers)",
  xlab = "Samples",
  ylab = "Abundance",
  las  = 2,                 # rotate x-axis labels
  cex.axis = 0.6            # shrink axis labels if many samples
)

```

```{r}
virome <-X_clean 
X <- as.matrix(virome)
if (any(X < 0, na.rm = TRUE)) stop("Input contains negative values.")
cs <- colSums(X, na.rm = TRUE)


#apply closure to 1
X <- sweep(X, 2, ifelse(cs == 0, NA, cs), "/")

# 2) zCompositions expects samples in ROWS -> transpose
X_t <- t(X)

X_t <-as.data.frame(X_t)
```

Perform Multiplicative replacement on the Virome dataset

```{r}


X_rep_t <- cmultRepl(
  X_t,
  label = 0,
  method = "CZM",
  output = "prop",
  z.warning = 1,      # effectively no warning threshold
  z.delete  = FALSE,  # do NOT delete high-zero columns
  suppress.print = TRUE, 
  adjust = TRUE
)
```

```{r}
X_rep <- t(X_rep_t)
X_rep <- sweep(X_rep, 2, colSums(X_rep), "/")

# Result:
virome_numeric_pc <- as.data.frame(X_rep)
```

```{r}


```

```{r}
# 4) VERIFY closure
col_sums <- colSums(virome_numeric_pc)
print(summary(col_sums))
cat("Min col sum:", min(col_sums), "\n")
cat("Max col sum:", max(col_sums), "\n")

tol <- 1e-12
offenders <- which(abs(col_sums - 1) > tol)
if (length(offenders) > 0) {
  cat("Columns not closed within tolerance:", paste(colnames(virome_numeric_pc)[offenders], collapse = ", "), "\n")
} else {
  cat("All columns sum to 1 within tolerance (", tol, ").\n", sep = "")
}
```

Apply CLR transformation on the viromic data

```{r}
gm_per_sample <- exp(colMeans(log(virome_numeric_pc)))

# CLR transform: log( value / column geometric mean )
#microbes_clr <- log(sweep(microbes_numeric_pc, 2, gm_per_sample, "/"))
virome_clr <- log(sweep(virome_numeric_pc, 2, gm_per_sample, "/"))

head(virome_clr)
```

```{r}
clr_sums <- colSums(virome_clr)

# Print results
for (sample in names(clr_sums)) {
  cat(sample, ": sum =", sprintf("%.6f", clr_sums[[sample]]), "\n")
}
```

```{r}
virome_clr_T <- t(virome_clr)

```

```{r}
virome <- virome_clr_T          # overwrite bacteriome with the processed data
# ensure it is a data.frame (it already is, but this is explicit)
virome <- as.data.frame(virome)
```

Save for mimenet

```{r}
write.table(
  virome,
  file = "~/data/virome_RA_mimenet.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)
```

Back to spls

```{r}
library(mixOmics)
set.seed(12345)

virome <- data.frame(lapply(virome, as.numeric))
metabolome <- data.frame(lapply(metabolome, as.numeric))

X <- data.frame(virome)
Y <- data.frame(metabolome)

rownames(X) <- NULL
rownames(Y) <- NULL

```

```{r}
num_cols <- sapply(metabolome, is.numeric)
metabolome[num_cols] <- lapply(metabolome[num_cols], function(x) log10(x + 1))


metabolome
```

```{r}

pca.virome <- pca(X, ncomp =10, scale = TRUE , center = TRUE)
pca.metabolites <-pca(Y, ncomp = 10, scale = TRUE, center= TRUE)
plot(pca.virome)


plot(pca.metabolites)
```

```{r}
pca.clr <- pca(X) # undergo PCA on CLR transformed data

plotIndiv(pca.clr,  # plot samples
          legend = TRUE)
```

```{r}
set.seed(12345)

#X<- scale(X,scale = TRUE, center= TRUE)
#Y <- scale(Y,scale = TRUE, center= TRUE)
pls.model <- pls(X = X, Y = Y, ncomp = 5, mode = 'regression', max.iter =1000, near.zero.var = TRUE, scale = TRUE)

png("Q2_plot_virome_metabolome_not_normalized.png", width = 12, height = 10, units = "in", res = 300)

Q2.pls2.model <- perf(pls.model, validation = 'Mfold', folds = 10, 
                      nrepeat = 10)
dev.off()
```

```{r}
Q2.pls2.model$measures$Q2.total
```

The Q2 mean of the pls model is 0.021647 ND STD0.0019. The signal is very limited

### The Q2 mean without PQN normalization in the PLS is 0.06025 with 0.0027

```{r}
ncomp <- 5
list.keepX <- c(seq(10, 50,5))
list.keepY <- c(seq(5 ,25,5))         

```

```{r}
library(BiocParallel)
set.seed(33)

tune.spls.model.trial <- tune.spls(X, Y,
                                   test.keepX = list.keepX,
                                   test.keepY = list.keepY,
                                   ncomp = 2,
                                   nrepeat = 10, folds = 10,
                                   mode = 'regression',
                                   measure = 'cor',max.iter=1000,near.zero.var = TRUE,scale = TRUE,
                                   BPPARAM = BiocParallel::SerialParam()
                                  
)
plot(tune.spls.model.trial)
```

```{r}
tune.spls.model.trial$choice.keepX
tune.spls.model.trial$choice.keepY
```

```{r}
choice.keepX <- tune.spls.model.trial$choice.keepX 

# extract optimal number of variables for Y datafram
choice.keepY <- tune.spls.model.trial$choice.keepY

optimal.ncomp <-  length(choice.keepX)

final.spls.model <- spls(X, Y, ncomp = 2, 
                    keepX = choice.keepX,
                    keepY = choice.keepY,scale = TRUE,
                    mode = "regression", near.zero.var = TRUE)

```

```{r}
final.spls.model$prop_expl_var

```

```{r}
png("~/code/Q2_plot_virome_pofiles_metabolome_no_normalization_spls.png", width = 12, height = 10, units = "in", res = 300)

set.seed(42)
pe <- perf(final.spls.model, validation = "Mfold", folds = 10, nrepeat = 10, progressBar = FALSE,near.zero.var = TRUE,scale = TRUE)
plot(pe, criterion = 'Q2.total')


dev.off()
```

Q2 mean for the sPLS model = 0.01742

without PQN

## for the sPLS I am getting 0.042 and 0.0054 std

```{r}
pe$measures$Q2.total
```

```{r}
selectVar(final.spls.model, comp = 1)$X$value

```

```{r}
stab.pe.comp1 <- pe$features$stability.X$comp1
# Averaged stability of the X selected features across CV runs, as shown in Table
stab.pe.comp1[1:choice.keepX[1]]

# We extract the stability measures of only the variables selected in spls2.liver
extr.stab.pe.comp1 <- stab.pe.comp1[selectVar(final.spls.model,                                                                  comp =1)$X$name]
```

```{Stability measure (occurence of selection) of the bottom 20 variables from X selected with sPLS2 across repeated 10-fold subsampling on component 1}

Virome metabolome

 Adrianviridae  Alexanderviridae     Amandaviridae      Bellaviridae     Hannahviridae 
             1.00              1.00              1.00              1.00              1.00 
    Haraldviridae     Mikkelviridae      Sisseviridae  Sylvesterviridae           VFC_426 
             1.00              1.00              1.00              1.00              1.00 
          VFC_486           VFC_523           VFC_617            VFC_68           VFC_421 
             1.00              1.00              1.00              1.00              0.99 
          VFC_468     Bertilviridae     Martinviridae       Alpaviridae  Mikkelineviridae 
             0.99              0.98              0.97              0.95              0.95 
          VFC_473           VFC_503      Edithviridae     Elviraviridae Kristofferviridae 
             0.95              0.95              0.93              0.93              0.92 
   Melanieviridae           VFC_627           VFC_393            VFC_85           VFC_378 
             0.90              0.87              0.85              0.84              0.84 
     Elenaviridae      Jeppeviridae    Gabrielviridae            VFC_82           VFC_476 
             0.78              0.76              0.68              0.68              0.65 
          VFC_304        Arnviridae     Agneteviridae     Dagmarviridae       Almaviridae 
             0.64              0.61              0.53              0.53              0.52 
```

```{r}
# Coerce loadings to numeric vector
xload_comp1 <- as.numeric(final.spls.model$loadings$X[, 1])
df <- data.frame(
  variable = rownames(final.spls.model$loadings$X),
  loading = xload_comp1
)

# Keep only nonzero loadings
df <- df[df$loading != 0 & !is.na(df$loading), ]

# Plot
p <- ggplot(df, aes(x = reorder(variable, loading), y = loading, fill = loading)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  labs(title = "Viral loadings- Comp 1", x = "Viral Families", y = "Loading") +
  theme_minimal() +
    theme(axis.text.y = element_text(size = 6))   # smaller font for variables
ggsave("spls_X_loadings_virome_not_normalization_level.pdf", plot = p, width = 8, height = 6)
p
```

```{r}
# Coerce loadings to numeric vector
yload_comp1 <- as.numeric(final.spls.model$loadings$Y[, 1])
df <- data.frame(
  variable = rownames(final.spls.model$loadings$Y),
  loading = yload_comp1
)

# Keep only nonzero loadings
df <- df[df$loading != 0 & !is.na(df$loading), ]

# Plot
p2<- ggplot(df, aes(x = reorder(variable, loading), y = loading, fill = loading)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red")  +
  labs(title = "Metabolites loadings - Comp 1", x = "Metabolites", y = "Loading") +
  theme_minimal() +
    theme(axis.text.y = element_text(size = 6))   
ggsave("spls_Y_loadings_comp1_metabolites_virome_families_level_nor_normalization.pdf", plot = p2, width = 8, height = 6)
```

```{r}
pdf("cim_plot_viral_families_metabolome_no_normalization.pdf", width = 24, height = 30,paper = "special", onefile = TRUE)

cim_res <- cim(
  final.spls.model,
  comp = 1:2,
  ylab = "Bacterial Species",
  xlab = "metabolites",
  color = colorRampPalette(c("blue", "white", "red"))(101),
  margins = c(10, 30)    # bottom, left — give labels room
)

dev.off()
```

```{r}
cor_mat <- cim_res$mat.cor

# convert to long-format table
cor_table <- as.data.frame(as.table(cor_mat))
colnames(cor_table) <- c("Virus_Families", "Metabolite", "Correlation")

# inspect or save
head(cor_table)
```

```{r}
write.csv(cor_table, "virome_metabolite_correlations.csv", row.names = FALSE)

```
