#!/usr/bin/env Rscript

# Apply the exact Tran per-document R multinomial draw and observed-vocabulary
# filtering to externally supplied graph-adapted W and exact-source A.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("usage: generate_tran_counts_from_factors.R A_CSV W_CSV OUTDIR COUNT_SEED N N_ANCHORS")
}

A_file <- normalizePath(args[[1]], mustWork = TRUE)
W_file <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- args[[3]]
count_seed <- as.integer(args[[4]])
N <- as.integer(args[[5]])
n_anchors <- as.integer(args[[6]])

A_full <- as.matrix(read.csv(A_file, check.names = FALSE))
W_document_by_topic <- as.matrix(read.csv(W_file, check.names = FALSE))
K <- ncol(W_document_by_topic)
if (ncol(A_full) != K) {
  stop("A and W topic dimensions differ")
}
if (max(abs(colSums(A_full) - 1)) > 1e-9) {
  stop("A columns must sum to one")
}
if (max(abs(rowSums(W_document_by_topic) - 1)) > 1e-9) {
  stop("W rows must sum to one")
}

population_full_feature_by_document <- A_full %*% t(W_document_by_topic)
set.seed(count_seed)
counts_feature_by_document <- sapply(
  seq_len(nrow(W_document_by_topic)),
  function(i) rmultinom(1, N, population_full_feature_by_document[, i])
)
observed <- which(rowSums(counts_feature_by_document) > 0)
counts <- t(counts_feature_by_document[observed, , drop = FALSE])
A_retained <- A_full[observed, , drop = FALSE]
A_retained <- A_retained %*% diag(1 / colSums(A_retained))
source_population <- t(population_full_feature_by_document[observed, , drop = FALSE])
factor_population <- W_document_by_topic %*% t(A_retained)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(counts, file.path(output_dir, "counts.csv"), row.names = FALSE)
write.csv(A_retained, file.path(output_dir, "A_feature_by_topic.csv"), row.names = FALSE)
write.csv(W_document_by_topic, file.path(output_dir, "W_document_by_topic.csv"), row.names = FALSE)
write.csv(source_population, file.path(output_dir, "source_population_document_by_word.csv"), row.names = FALSE)
write.csv(factor_population, file.path(output_dir, "factor_population_document_by_word.csv"), row.names = FALSE)
write.csv(data.frame(vocab_index_r = observed), file.path(output_dir, "vocabulary.csv"), row.names = FALSE)

anchor_indices <- if (n_anchors > 0) seq_len(K * n_anchors) else integer(0)
anchor_survival <- anchor_indices %in% observed
writeLines(
  c(
    "count_generator=R_stats_rmultinom_per_document",
    paste0("count_seed=", count_seed),
    paste0("N=", N),
    paste0("n=", nrow(W_document_by_topic)),
    paste0("requested_p=", nrow(A_full)),
    paste0("observed_p=", length(observed)),
    paste0("K=", K),
    paste0("n_anchors=", n_anchors),
    paste0("anchor_indices_r=", paste(anchor_indices, collapse = ",")),
    paste0("anchor_survival=", paste(as.integer(anchor_survival), collapse = ",")),
    paste0("source_D0_min_row_mass=", min(rowSums(source_population))),
    paste0("source_D0_max_row_mass=", max(rowSums(source_population)))
  ),
  file.path(output_dir, "count_metadata.txt")
)
