#!/usr/bin/env Rscript

# Freeze one exact local Tran mixed-decay draw and its native Topic-SCORE fit.
# Run from the root of the local topic-modeling repository so score.r can find
# its own helper sources.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: generate_tran_mixed_topicscore_fixture.R TOPIC_MODELING_ROOT OUTPUT_DIR")
}

topic_root <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- if (grepl("^/", args[[2]])) {
  args[[2]]
} else {
  file.path(getwd(), args[[2]])
}
generator_file <- file.path(
  topic_root, "r", "experiments", "synthetic", "synthetic_dataset.R"
)

suppressPackageStartupMessages(library(VGAM))
suppressPackageStartupMessages(library(Matrix))

# Evaluate only the authoritative generator definition, not the research
# script's top-level library/source side effects.
expressions <- parse(file = generator_file, keep.source = TRUE)
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
  stop("synthetic_dataset_creation was not found")
}

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(topic_root)
source(file.path("r", "score.r"))

settings <- list(
  seed = 4132L,
  n = 36L,
  p = 80L,
  N = 60L,
  K = 3L,
  alpha_dirichlet = 1,
  n_anchors = 2L,
  delta_anchor = 0.01,
  a_zipf = 1,
  offset_zipf = 2.7
)

data <- do.call(
  synthetic_dataset_creation,
  c(settings, list(vary_by_topic = FALSE, sparsity = TRUE))
)
D_word_by_document <- t(data$D) / settings$N
set.seed(settings$seed)
fit <- score(
  D = D_word_by_document,
  K = settings$K,
  normalize = "norm",
  Mquantile = 0,
  VHMethod = "SP",
  returnW = TRUE,
  estimateK = FALSE,
  as.sparse = FALSE
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data$D, file.path(output_dir, "counts_document_by_word.csv"), row.names = FALSE)
write.csv(data$A, file.path(output_dir, "A_feature_by_topic.csv"), row.names = FALSE)
write.csv(data$W, file.path(output_dir, "W_topic_by_document.csv"), row.names = FALSE)
write.csv(data$D0, file.path(output_dir, "population_document_by_word.csv"), row.names = FALSE)
write.csv(data.frame(vocab_index_r = data$vocab), file.path(output_dir, "vocabulary.csv"), row.names = FALSE)
write.csv(as.matrix(fit$A_hat), file.path(output_dir, "topicscore_A_feature_by_topic.csv"), row.names = FALSE)
write.csv(as.matrix(fit$W_hat), file.path(output_dir, "topicscore_W_topic_by_document.csv"), row.names = FALSE)
write.csv(as.matrix(fit$R), file.path(output_dir, "topicscore_ratio_cloud.csv"), row.names = FALSE)
write.csv(as.matrix(fit$V), file.path(output_dir, "topicscore_vertices.csv"), row.names = FALSE)
write.csv(as.matrix(fit$Pi), file.path(output_dir, "topicscore_Pi.csv"), row.names = FALSE)

anchor_indices_r <- seq_len(settings$K * settings$n_anchors)
anchor_survives <- anchor_indices_r %in% data$vocab
metadata <- c(
  paste0("generator_source=", normalizePath(generator_file)),
  paste0("topicscore_source=", normalizePath(file.path(topic_root, "r", "score.r"))),
  "topicscore_normalize=norm",
  "topicscore_Mquantile=0",
  "topicscore_VHMethod=SP",
  "topicscore_as_sparse=false",
  paste0("seed=", settings$seed),
  paste0("n=", settings$n),
  paste0("requested_p=", settings$p),
  paste0("observed_p=", nrow(data$A)),
  paste0("N=", settings$N),
  paste0("K=", settings$K),
  paste0("alpha_dirichlet=", settings$alpha_dirichlet),
  paste0("n_anchors=", settings$n_anchors),
  paste0("delta_anchor=", settings$delta_anchor),
  paste0("a_zipf=", settings$a_zipf),
  paste0("offset_zipf=", settings$offset_zipf),
  paste0("anchor_indices_r=", paste(anchor_indices_r, collapse = ",")),
  paste0("anchor_survival=", paste(as.integer(anchor_survives), collapse = ",")),
  paste0("count_row_sum_min=", min(rowSums(data$D))),
  paste0("count_row_sum_max=", max(rowSums(data$D)))
)
writeLines(metadata, file.path(output_dir, "metadata.txt"))
