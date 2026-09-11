#!/usr/bin/env Rscript

# Cross-language fixture for the exact threshold block in topic-modeling/r/score.r.
# The source tree is never modified; all outputs are written beside this script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: generate_tran_threshold_fixture.R TOPIC_MODELING_SOURCE [OUTPUT_DIR]")
source_root <- normalizePath(args[[1]])
output_dir <- if (length(args) >= 2) args[[2]] else dirname(normalizePath(sys.frame(1)$ofile))

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(source_root)
source("r/score.r")

counts <- rbind(
  c(8, 7, 6, 7, 1, 1, 2, 1),
  c(1, 1, 2, 1, 8, 7, 6, 7),
  c(4, 4, 4, 4, 4, 4, 4, 4),
  c(3, 3, 3, 3, 3, 3, 3, 3),
  c(1, 1, 1, 1, 1, 1, 1, 1),
  c(1, 1, 1, 1, 1, 1, 1, 1),
  c(1, 1, 1, 1, 1, 1, 1, 1),
  c(1, 1, 1, 1, 1, 1, 1, 1),
  c(0, 1, 1, 1, 0, 1, 1, 1),
  c(0, 0, 0, 0, 0, 0, 0, 0)
)
stopifnot(all(colSums(counts) == 20))

D <- counts / 20
p <- nrow(D)
n <- ncol(D)
N <- 20
alpha <- 0.5

# This block is intentionally identical to score.r, including strict inequality
# and the top-10-percent fallback.
M <- as.numeric(rowMeans(D))
threshold_J <- alpha * sqrt(log(max(p, n)) / (N * n))
setJ <- which(M > threshold_J)
fallback_active <- FALSE
if (length(setJ) < 0.1 * length(M)) {
  fallback_active <- TRUE
  setJ <- sort(M, decreasing = TRUE, index.return = TRUE)$ix[
    1:ceiling(0.1 * length(M))
  ]
}
newD <- D[setJ, , drop = FALSE]

# Run the original function as well.  Its returned thresholded fraction confirms
# that the source call took the same feature-selection branch.
fit <- score(
  D = D,
  K = 2,
  N = N,
  threshold = TRUE,
  alpha = alpha,
  normalize = "TTS",
  VHMethod = "SP",
  estimateK = FALSE,
  as.sparse = FALSE
)
stopifnot(isTRUE(all.equal(fit$thresholded, 1 - length(setJ) / p)))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(
  data.frame(
    feature_index_r = seq_len(p),
    eta_hat = M,
    retained = seq_len(p) %in% setJ
  ),
  file.path(output_dir, "tran_threshold_eta.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(feature_index_r = setJ, newD, check.names = FALSE),
  file.path(output_dir, "tran_threshold_matrix.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(feature_index_r = seq_len(p), fit$A_hat, check.names = FALSE),
  file.path(output_dir, "tran_score_A_hat.csv"),
  row.names = FALSE
)
writeLines(
  c(
    paste0("source_root=", source_root),
    paste0("alpha=", format(alpha, digits = 17)),
    paste0("N=", N),
    paste0("n=", n),
    paste0("p=", p),
    paste0("threshold=", format(threshold_J, digits = 17)),
    paste0("retained_indices_r=", paste(setJ, collapse = ",")),
    paste0("fallback_active=", tolower(as.character(fallback_active))),
    paste0("score_thresholded_fraction=", format(fit$thresholded, digits = 17))
  ),
  file.path(output_dir, "tran_threshold_metadata.txt")
)

cat("TRAN_THRESHOLD_FIXTURE_OK\n")
cat(sprintf("threshold=%.17g retained=%s\n", threshold_J, paste(setJ, collapse = ",")))
