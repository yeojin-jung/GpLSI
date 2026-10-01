#!/usr/bin/env Rscript

# Generate a reduced pilot data set by evaluating the authoritative
# synthetic_dataset_creation definition directly from the local Tran source.
# Other top-level expressions in that research script are deliberately not run.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12) {
  stop(paste(
    "usage: generate_tran_pilot_data.R SOURCE OUTDIR SEED n p N K",
    "alpha_dirichlet n_anchors delta_anchor a_zipf offset_zipf"
  ))
}

source_file <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
seed <- as.integer(args[[3]])
n <- as.integer(args[[4]])
p <- as.integer(args[[5]])
N <- as.integer(args[[6]])
K <- as.integer(args[[7]])
alpha_dirichlet <- as.numeric(args[[8]])
n_anchors <- as.integer(args[[9]])
delta_anchor <- as.numeric(args[[10]])
a_zipf <- as.numeric(args[[11]])
offset_zipf <- as.numeric(args[[12]])

suppressPackageStartupMessages(library(VGAM))
expressions <- parse(file = source_file, keep.source = TRUE)
found <- FALSE
for (expression in expressions) {
  if (
    is.call(expression) && identical(as.character(expression[[1]]), "<-") &&
    identical(as.character(expression[[2]]), "synthetic_dataset_creation")
  ) {
    eval(expression, envir = .GlobalEnv)
    found <- TRUE
    break
  }
}
if (!found) {
  stop("synthetic_dataset_creation was not found in the authoritative source")
}

data <- synthetic_dataset_creation(
  n = n,
  K = K,
  p = p,
  alpha_dirichlet = alpha_dirichlet,
  a_zipf = a_zipf,
  offset_zipf = offset_zipf,
  n_anchors = n_anchors,
  delta_anchor = delta_anchor,
  N = N,
  seed = seed,
  vary_by_topic = FALSE,
  sparsity = TRUE
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data$D, file.path(output_dir, "counts.csv"), row.names = FALSE)
write.csv(data$A, file.path(output_dir, "A_feature_by_topic.csv"), row.names = FALSE)
write.csv(data$W, file.path(output_dir, "W_topic_by_document.csv"), row.names = FALSE)
write.csv(data$D0, file.path(output_dir, "population_document_by_feature.csv"), row.names = FALSE)
write.csv(data.frame(vocab_index_r = data$vocab), file.path(output_dir, "vocabulary.csv"), row.names = FALSE)

metadata <- c(
  paste0("source=", source_file),
  paste0("seed=", seed),
  paste0("n=", n),
  paste0("requested_p=", p),
  paste0("observed_p=", nrow(data$A)),
  paste0("N=", N),
  paste0("K=", K),
  paste0("alpha_dirichlet=", alpha_dirichlet),
  paste0("n_anchors=", n_anchors),
  paste0("delta_anchor=", delta_anchor),
  paste0("a_zipf=", a_zipf),
  paste0("offset_zipf=", offset_zipf)
)
writeLines(metadata, file.path(output_dir, "metadata.txt"))
