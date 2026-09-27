#!/usr/bin/env Rscript

# Fixed-center and fixed-point-cloud parity fixtures for MixedSCORE SVS.

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1) args[[1]] else normalizePath(".")
output_dir <- if (length(args) >= 2) args[[2]] else file.path(repo_root, "tests", "fixtures")
.libPaths(c(file.path(repo_root, "external_references", "Rlib"), .libPaths()))
source(file.path(repo_root, "external_references", "MixedSCORE", "mixedSCORE.R"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fixed_cases <- list(
  k3 = list(
    K = 3,
    centers = rbind(
      c(0, 0), c(1, 0), c(0, 1), c(0.2, 0.2), c(0.6, 0.2), c(0.2, 0.6)
    )
  ),
  k2 = list(K = 2, centers = matrix(c(-1, 1, -0.2, 0.3), ncol = 1))
)

for (case_name in names(fixed_cases)) {
  case <- fixed_cases[[case_name]]
  result <- vertexSearch(case$centers, case$K)
  write.csv(
    data.frame(center_index_r = seq_len(nrow(case$centers)), case$centers),
    file.path(output_dir, paste0("mixedscore_svs_", case_name, "_centers.csv")),
    row.names = FALSE
  )
  write.csv(
    data.frame(selected_index_r = result$ind, result$vertices),
    file.path(output_dir, paste0("mixedscore_svs_", case_name, "_result.csv")),
    row.names = FALSE
  )
  writeLines(
    c(
      paste0("K=", case$K),
      paste0("objective=", format(result$dist, digits = 17)),
      paste0("candidate_count=", choose(nrow(case$centers), case$K))
    ),
    file.path(output_dir, paste0("mixedscore_svs_", case_name, "_metadata.txt"))
  )
}

# Save every R-generated center matrix so the adaptive selection can be checked
# independently of cross-language k-means differences.
R <- rbind(
  c(0.00, 0.00), c(0.03, 0.02), c(-0.02, 0.04),
  c(1.00, 0.00), c(0.96, 0.04), c(1.03, -0.02),
  c(0.00, 1.00), c(0.04, 0.96), c(-0.03, 1.02),
  c(0.25, 0.25), c(0.50, 0.20), c(0.20, 0.55),
  c(0.35, 0.35), c(0.10, 0.45), c(0.45, 0.10)
)
K <- 3
set.seed(2024)
L_candidates <- (K + 1):(3 * K)
out_list <- lapply(L_candidates, function(L) {
  centers <- kmeans(R, L, iter.max = 100, nstart = 100)$centers
  result <- vertexSearch(centers, K = K)
  write.csv(
    data.frame(center_index_r = seq_len(L), centers),
    file.path(output_dir, paste0("mixedscore_adaptive_centers_L", L, ".csv")),
    row.names = FALSE
  )
  result
})

centers_K <- kmeans(R, K, iter.max = 100, nstart = 100)$centers
write.csv(
  data.frame(center_index_r = seq_len(K), centers_K),
  file.path(output_dir, paste0("mixedscore_adaptive_centers_L", K, ".csv")),
  row.names = FALSE
)

deltas <- sapply(seq_along(L_candidates), function(i) {
  previous <- if (i == 1) centers_K else out_list[[i - 1]]$vertices
  current <- out_list[[i]]$vertices
  perms <- combinat::permn(seq_len(K))
  delta <- min(sapply(perms, function(order) {
    max(rowSums(current[order, , drop = FALSE] - previous)^2)
  }))
  delta / (1 + out_list[[i]]$dist)
})
selected_position <- which.min(deltas)
selected_L <- L_candidates[selected_position]
write.csv(
  data.frame(
    L = L_candidates,
    objective = sapply(out_list, function(value) value$dist),
    stability = deltas,
    candidate_count = sapply(L_candidates, function(L) choose(L, K))
  ),
  file.path(output_dir, "mixedscore_adaptive_summary.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(point_index_r = seq_len(nrow(R)), R),
  file.path(output_dir, "mixedscore_adaptive_point_cloud.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    selected_index_r = out_list[[selected_position]]$ind,
    out_list[[selected_position]]$vertices
  ),
  file.path(output_dir, "mixedscore_adaptive_selected_vertices.csv"),
  row.names = FALSE
)
writeLines(
  c(
    "seed=2024",
    paste0("K=", K),
    paste0("selected_L=", selected_L),
    "kmeans_iter_max=100",
    "kmeans_nstart=100"
  ),
  file.path(output_dir, "mixedscore_adaptive_metadata.txt")
)

# Verify the saved expansion is the source function, not a reinterpretation.
set.seed(2024)
official <- vertexHunting(R, K)
stopifnot(official$L == selected_L)
stopifnot(isTRUE(all.equal(official$vertices, out_list[[selected_position]]$vertices)))
stopifnot(isTRUE(all.equal(official$centers, out_list[[selected_position]]$centers)))

cat("MIXEDSCORE_SVS_FIXTURE_OK\n")
cat(sprintf("selected_L=%d\n", selected_L))
