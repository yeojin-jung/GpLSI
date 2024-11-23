get_K = function(x){
  v = seq(1:length(x))
  diff = x-v
  K = length(x)
  for(i in 1:length(x)){
    if(diff[i]!=0){
      K=x[i]
      break
    }
  }
  return(K)
}

get_matrices = function(ntopics, immune_names, root_path_Ahat, root_path_What){
    filenames_A = list.files(root_path_Ahat)
    filenames_W = list.files(root_path_What)
    models = list()
    for(t in 1:ntopics){
      topic_name = as.character(t)
      Ahat = read.csv(paste0(root_path_Ahat, '/', filenames_A[t]), header = FALSE)
      Ahat = t(Ahat)
      rownames(Ahat) = immune_names
      What = read.csv(paste0(root_path_What, '/', filenames_W[t]), header = FALSE)
      What = as.matrix(What)
      models[[topic_name]] <- list(gamma = What, beta = t(as.matrix(Ahat)))
    }
    return(models)
}

get_paths = function(ntopics, immune_names, root_path_Ahat, root_path_What, plot_path=NULL){
  models = get_matrices(ntopics, immune_names, root_path_Ahat, root_path_What)
  result = align_topics(models)
  paths = compute_number_of_paths(result, plot = FALSE)$n_paths
  K_hat = get_K(paths)
  print(paste0("K_hat is ", K_hat))
  p = plot(result)
  if(!is.null(plot_path)){
    ggsave(filename = plot_path, plot = p) 
  }
  print(p)
}
