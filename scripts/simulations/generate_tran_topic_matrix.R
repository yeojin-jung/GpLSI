#!/usr/bin/env Rscript

# Generate the Tran synthetic topic matrix before count-dependent vocabulary
# deletion.  The authoritative function is parsed from the supplied source and
# stopped immediately after D0 is formed, before its multinomial count draw.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11) {
  stop(paste(
    "usage: generate_tran_topic_matrix.R SOURCE OUTPUT SEED n p K",
    "alpha_dirichlet n_anchors delta_anchor a_zipf offset_zipf"
  ))
}

source_file <- normalizePath(args[[1]], mustWork = TRUE)
output_file <- args[[2]]
seed <- as.integer(args[[3]])
n <- as.integer(args[[4]])
p <- as.integer(args[[5]])
K <- as.integer(args[[6]])
alpha_dirichlet <- as.numeric(args[[7]])
n_anchors <- as.integer(args[[8]])
delta_anchor <- as.numeric(args[[9]])
a_zipf <- as.numeric(args[[10]])
offset_zipf <- as.numeric(args[[11]])

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

source_function <- synthetic_dataset_creation
body_items <- as.list(body(source_function))[-1]
is_count_draw <- vapply(
  body_items,
  function(item) {
    is.call(item) && identical(as.character(item[[1]]), "<-") &&
      identical(as.character(item[[2]]), "X")
  },
  logical(1)
)
if (sum(is_count_draw) != 1) {
  stop("expected exactly one top-level assignment to X in the Tran generator")
}
count_index <- which(is_count_draw)[[1]]
truncated_body <- c(
  body_items[seq_len(count_index - 1)],
  list(quote(return(list(A = A, W = W, D0 = D0))))
)
body(source_function) <- as.call(c(as.name("{"), truncated_body))

data <- source_function(
  n = n,
  K = K,
  p = p,
  alpha_dirichlet = alpha_dirichlet,
  a_zipf = a_zipf,
  offset_zipf = offset_zipf,
  n_anchors = n_anchors,
  delta_anchor = delta_anchor,
  N = 1,
  seed = seed,
  vary_by_topic = FALSE,
  sparsity = TRUE
)
A <- data$A

stopifnot(all(dim(A) == c(p, K)))
stopifnot(max(abs(colSums(A) - 1)) < 1e-10)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(A, output_file, row.names = FALSE)
writeLines(
  c(
    paste0("source=", source_file),
    "source_function=synthetic_dataset_creation",
    "adaptation=truncate_authoritative_function_before_X_count_draw",
    paste0("seed=", seed),
    paste0("n=", n),
    paste0("p=", p),
    paste0("K=", K),
    paste0("alpha_dirichlet=", alpha_dirichlet),
    paste0("n_anchors=", n_anchors),
    paste0("delta_anchor=", delta_anchor),
    paste0("a_zipf=", a_zipf),
    paste0("offset_zipf=", offset_zipf)
  ),
  paste0(output_file, ".metadata.txt")
)
