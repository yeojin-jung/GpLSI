library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

# Function to reshape and process data
process_data <- function(data, error_cols, method_names) {
  data_long <- data %>%
    pivot_longer(cols = all_of(error_cols), names_to = "method_error", values_to = "error") %>%
    mutate(Method = method_names[match(method_error, error_cols)]) %>%
    mutate(Method = factor(Method, levels = method_names))
  return(data_long)
}

generate_plot <- function(data, x_var, y_var, group_vars, 
                           x_label, y_label, facet_var = NULL, 
                           facet_labels = NULL, log_scale = FALSE,
                           tilt = TRUE) {
  # Base ggplot
  p <- ggplot(data, aes_string(x = x_var, y = y_var, color = "Method", group = "Method")) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes_string(ymin = paste0(y_var, "-sd_error"), ymax = paste0(y_var, "+sd_error")), width = 0.2) +
    labs(x = x_label, y = y_label) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 10),
      axis.title.x = element_text(size = 10),
      axis.title.y = element_text(size = 10),
      axis.text = element_text(size = 10)
    )
  if (tilt) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  if (!is.null(facet_var)) {
    p <- p + facet_wrap(as.formula(paste0("~ ", facet_var)), labeller = as_labeller(facet_labels), ncol = 3)
  }
  if (log_scale) {
    p <- p + scale_x_log10(breaks = unique(data[[x_var]]))
    }else{
      p <- p + scale_x_continuous(breaks = unique(data[[x_var]]))
    }
  return(p)
}

plot_everything = function(res_df, error_columns, method_names){
  plots <- list()
  res_df_long <- process_data(res_df, error_columns, method_names)
  
  # Error by (N x p)
  res_df_long1 <- res_df_long %>%
    filter(K == 3, n == 1000) %>%
    mutate(error = error / n) %>%
    group_by(N, p, Method) %>%
    summarize(
      mean_error = median(error, na.rm = TRUE),
      sd_error = IQR(error, na.rm = TRUE) / 2,
      .groups = 'drop'
    )
  
  p_labs <- paste("p =", c(20, 50, 100, 200))
  names(p_labs) <- unique(res_df_long1$p)
  plots[["plot_Np"]] <- generate_plot(res_df_long1, "N", "mean_error", c("N", "p", "Method"), "N", "l2 Error", "p", p_labs, log_scale = TRUE)
  
  # Plot 2
  res_df_long2 <- res_df_long %>%
    filter(K == 3, p == 20, N == 30) %>%
    mutate(adjusted_error = error / n) %>%
    group_by(n, Method) %>%
    summarize(
      mean_error = median(adjusted_error, na.rm = TRUE),
      sd_error = IQR(adjusted_error, na.rm = TRUE) / 2,
      .groups = 'drop'
    )
  
  plots[["plot_n"]] <- generate_plot(res_df_long2, "n", "mean_error", c("n", "Method"), "n", "l2 Error", tilt = FALSE)
  
  # Plot 3
  res_df_long3 <- res_df_long %>%
    filter(N == 30, p == 20, n == 1000) %>%
    mutate(error = error / n) %>%
    group_by(K, Method) %>%
    summarize(
      mean_error = median(error, na.rm = TRUE),
      sd_error = IQR(error, na.rm = TRUE) / 2,
      .groups = 'drop'
    )
  
  plots[["plot_K"]] <- generate_plot(res_df_long3, "K", "mean_error", c("K", "Method"), "K", "l2 Error", tilt = FALSE)
  return(plots)
}

# Read in data
model_root = file.path(getwd(), 'simulation/results_final')
files = list.files(model_root)
res_df = vroom(files)

method_names = c("pLSI", "GpLSI", "GpLSI onestep", "GpLSI XTX", "TopicSCORE", "LDA", "SLDA")

# Errors of W
error_columns <- c("plsi_err", "gplsi_err", "gplsi_onestep_err", "gplsi_XTX_err", "ts_err", "lda_err", "slda_err")
plots_l2_W = plot_everything(res_df, error_columns, method_names)
plots_l2_W[['plot_Np']]
plots_l2_W[['plot_n']]
plots_l2_W[['plot_K']]

error_columns <- c("plsi_l1_err", "gplsi_l1_err", "gplsi_onestep_l1_err", "gplsi_XTX_l1_err", "ts_l1_err", "lda_l1_err", "slda_l1_err")
plots_l1_W = plot_everything(res_df, error_columns, method_names)
plots_l1_W[['plot_Np']]
plots_l1_W[['plot_n']]
plots_l1_W[['plot_K']]

# Errors of A
error_columns <- c("A_plsi_err", "A_gplsi_err", "A_gplsi_onestep_err", "A_gplsi_XTX_err", "A_ts_err", "A_lda_err", "A_slda_err")
plots_l2_A = plot_everything(res_df, error_columns, method_names)
plots_l2_A[['plot_Np']]
plots_l2_A[['plot_n']]
plots_l2_A[['plot_K']]

error_columns <- c("A_plsi_l1_err", "A_gplsi_l1_err", "A_gplsi_onestep_l1_err", "A_gplsi_XTX_l1_err", "A_ts_l1_err", "A_lda_l1_err", "A_slda_l1_err")
plots_l1_A = plot_everything(res_df, error_columns, method_names)
plots_l1_A[['plot_Np']]
plots_l1_A[['plot_n']]
plots_l1_A[['plot_K']]

# Time
error_columns <- c("plsi_time", "gplsi_time", "gplsi_onestep_time", "gplsi_XTX_time", "ts_time", "lda_time", "slda_time")
plots_l2_W = plot_everything(res_df, error_columns, method_names)
plots_l2_W[['plot_Np']]
plots_l2_W[['plot_n']]
plots_l2_W[['plot_K']]

# Choice of regularization as N increases
res_df1 =  res_df %>%
  filter(K==3, n==1000, p==20) %>%
  select(N, opt_gamma) %>%
  group_by(N) %>%
  summarize(
    mean_error = median(opt_gamma, na.rm = TRUE),
    sd_error = IQR(opt_gamma, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p11 <- ggplot(res_df1, aes(x = N, 
                           y = mean_error)) +
  geom_point(color='#F8766D') +
  geom_line(color='#F8766D') +
  geom_errorbar(aes(ymin = mean_error - sd_error, ymax = mean_error + sd_error), width = 0.2,
                color='#F8766D') +
  labs(x = "N", y = expression(hat(rho)[MST-CV])) +
  theme_bw() +
  scale_x_log10()+
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10)  # Set axis text font size
  ) +
  scale_x_log10(breaks = unique(res_df1$N)) +  # Show labels only for points
  theme(legend.position = 'None')



# Graph Smoothness
model_root = file.path(getwd(), 'simulation/ncluster')
files = list.files(model_root)
res_cluster = vroom(files)

res_df_long <- res_cluster %>%
  pivot_longer(cols = c(plsi_err, hooi_err),
               names_to = "method_error", values_to = "error") %>%
  mutate(Method = case_when(
    str_detect(method_error, "^plsi_err$") ~ "pLSI",
    str_detect(method_error, "^hooi_err$") ~ "GpLSI"
  )) 
res_df_long$Method <- factor(res_df_long$Method, levels = c("GpLSI", "pLSI"))


res_df_long1 =  res_df_long %>%
  filter(K==5, ncluster<500, ncluster>10)%>%
  mutate(error = error / n) %>%
  group_by(ncluster, Method) %>%
  summarize(
    mean_error = median(error, na.rm = TRUE),
    sd_error = IQR(error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p_cluster <- ggplot(res_df_long1, aes(x = ncluster, 
                                      y = mean_error, 
                                      color = Method)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error + sd_error), width = 0.2) +
  labs(x = "Graph smoothness", y = "l2 Error") +
  theme_bw() +
  scale_x_log10()+
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10)  # Set axis text font size
  ) +
  scale_x_log10(breaks = unique(res_df_long1$ncluster))+
  theme(legend.position = 'None')






