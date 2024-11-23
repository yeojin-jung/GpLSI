library(readr)
library(ggplot2)
library(pROC)
library(compositions)
library(dplyr)
library(survival)

process_data_prop = function(file_path, nruns=10) {
  survival <- read_csv(file_path)
  colnames(survival) <- c("region_id", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "status", "recurrence", "time")
  
  #X <- scale(clr(survival[, 2:7]))
  X <- scale(ilr(survival[, 2:7]), center = TRUE, scale = FALSE)
  y <- as.factor(survival$recurrence)
  
  auc_values = numeric(nruns)
  roc_data_list = list()
  
  sensitivity_grid <- seq(0, 1, length.out = 26)
  
  for(i in 1:nruns){
    seed_num = 100+10*i
    train_index <- sample(1:nrow(survival), round(0.7 * nrow(survival)))
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    fit <- glm(y_train ~ X_train, family = binomial)
    test_data <- data.frame(X_train = I(X_test))
    test_probs <- predict(fit, newdata = test_data, type = "response")
    
    roc_obj <- roc(y_test, test_probs)
    auc_values[i] = auc(roc_obj)
    
    out <- approx(roc_obj$sensitivities, roc_obj$specificities, xout = sensitivity_grid)
    
    roc_data_list[[i]] <- data.frame(
      specificities = out$y,
      sensitivities = out$x
    )
  }
  mean_auc <- mean(auc_values)
  
  roc_df = do.call(rbind, roc_data_list)
  roc_df_mean = roc_df %>%
    group_by(sensitivities) %>%
    summarise(specificities = mean(specificities),
              sensitivities = mean(sensitivities))
  
  return(list(roc = roc_df_mean, auc = mean_auc))
}


process_data_binary <- function(file_path, nruns=10) {
  survival <- read_csv(file_path)
  colnames(survival) <- c("region_id", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "status", "recurrence", "time")
  
  variables = paste0('Topic',1:6)
  res.cut = surv_cutpoint(
    survival,
    time = "time",
    event = "recurrence",
    variables,
    minprop = 0.1,
    progressbar = TRUE
  )
  
  res.cat <- surv_categorize(res.cut)
  res.cat$Topic1 <- as.factor(res.cat$Topic1)
  res.cat$Topic2 <- as.factor(res.cat$Topic2)
  res.cat$Topic3 <- as.factor(res.cat$Topic3)
  res.cat$Topic4 <- as.factor(res.cat$Topic4)
  res.cat$Topic5 <- as.factor(res.cat$Topic5)
  res.cat$Topic6 <- as.factor(res.cat$Topic6)
  
  X <- model.matrix(~ Topic1 + Topic2 + Topic3 + Topic4 + Topic5 + Topic6, data = res.cat)[, -1]
  y <- as.factor(survival$recurrence)
  
  auc_values = numeric(nruns)
  roc_data_list = list()
  
  sensitivity_grid <- seq(0, 1, length.out = 26)
  
  for(i in 1:nruns){
    seed_num = 100+10*i
    set.seed(seed_num)
    train_index <- sample(1:nrow(survival), round(0.7 * nrow(survival)))
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    fit <- glm(y_train ~ ., data = as.data.frame(X_train), family = binomial)
    test_data <- as.data.frame(X_test)
    test_probs <- predict(fit, newdata = test_data, type = "response")
    
    roc_obj <- roc(y_test, test_probs)
    auc_values[i] = auc(roc_obj)
    
    out <- approx(roc_obj$sensitivities, roc_obj$specificities, xout = sensitivity_grid)
    
    roc_data_list[[i]] <- data.frame(
      specificities = out$y,
      sensitivities = out$x
    )
  }
  mean_auc <- mean(auc_values)
  
  roc_df = do.call(rbind, roc_data_list)
  roc_df_mean = roc_df %>%
    group_by(sensitivities) %>%
    summarise(specificities = mean(specificities),
              sensitivities = mean(sensitivities))
  return(list(roc = roc_df_mean, auc = mean_auc))
}

plot_roc = function(func, data_paths, method='proportion', nruns){
  method_names = c('GpLSI', 'pLSI', 'LDA')
  results <- lapply(data_paths, func, nruns)
  
  roc_objects <- lapply(results, function(res) res$roc)
  auc_scores <- sapply(results, function(res) res$auc)
  
  roc_data <- do.call(rbind, lapply(seq_along(roc_objects), function(i) {
    data.frame(
      FPR = 1 - roc_objects[[i]]$specificities,
      TPR = roc_objects[[i]]$sensitivities,
      Method = method_names[i],
      AUC = auc_scores[i]
    )
  }))
  roc_data <- roc_data %>%
    mutate(Method = factor(Method, levels = c("GpLSI", "pLSI", "LDA"), labels = c("GpLSI", "pLSI", "LDA")))
  
  # Plot ROC curves
  if(method == 'binary'){
    title = 'ROC Curves (Dichotomized)'
  }else{
    title = 'ROC Curves (Proportion)'
  }
  p = ggplot(roc_data, aes(x = FPR, y = TPR, color = Method)) +
    geom_line() +
    geom_abline(linetype = "dashed", color = "gray") +
    labs(title = NULL,
         x = "False Positive Rate",
         y = "True Positive Rate") +
    #annotate("text", x = 0.5, y = 0.3, label = paste("AUC (GpLSI):", round(auc_scores[1], 2)), color = "red", hjust = 0, size = 5) +
    #annotate("text", x = 0.5, y = 0.2, label = paste("AUC (pLSI):", round(auc_scores[2], 2)), color = "green", hjust = 0, size = 5) +
    #annotate("text", x = 0.5, y = 0.1, label = paste("AUC (LDA):", round(auc_scores[3], 2)), color = "blue", hjust = 0, size = 5) +
    theme_classic(base_size = 10) +  # Set consistent base font size
    theme(
      plot.title = element_text(size = 10),  # Set title font size
      axis.title.x = element_text(size = 10),  # Set X-axis title font size
      axis.title.y = element_text(size = 10),  # Set Y-axis title font size
      axis.text = element_text(size = 10),  # Set axis text font size
      legend.position = 'None'
    )
  print(p)
  return(roc_data)
}


plot_coefs = function(file_path, ntopics){
  survival = read_csv(file_path)
  colnames(survival) <- c("region_id", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "status", "time")
  # run logistic regression
  X = scale(ilr(survival[,2:(ntopics+1)]), center = TRUE, scale=FALSE)
  y = survival$status
  fit = glm(y ~ X, family=binomial)
  coef_logistic = ilr2clr(coef(fit)[-1],x=X)
  
  # run cox proportional hazards
  time = survival$time
  status = survival$status
  cox = coxph(Surv(time, status) ~ X)
  coef_cox = ilr2clr(coef(cox),x=X)
  
  # plot
  coef_data <- data.frame(
    Variable = factor(rep(paste0("Topic", 1:6), 2), levels = paste0("Topic", 6:1)),
    Coefficient = c(exp(coef_logistic), exp(coef_cox)),
    Model = rep(c("Logistic", "Cox PH"), each = 6)
  )
  p = ggplot(coef_data, aes(x = Coefficient, y = Variable, color = Model)) +
    geom_point(size = 3, aes(shape=Model)) +
    geom_vline(xintercept = 1, linetype = "dotted", color = "red") +
    theme_minimal() +
    coord_cartesian(xlim = c(0.8,1.2))+
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = "bottom") +
    labs(title = NULL,
         x = "Coefficient Value",
         y = NULL,
         color = "Model")+ 
    theme(plot.margin = margin(15, 15, 15, 15))
  return(p)
}


plot_KM = function(file_path){
  survival <- read_csv(file_path)
  colnames(survival) <- c("region_id", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "status", "recurrence", "time")
  
  variables = paste0('Topic',1:6)
  res.cut = surv_cutpoint(
    survival,
    time = "time",
    event = "recurrence",
    variables,
    minprop = 0.1,
    progressbar = TRUE
  )
  
  res.cat <- surv_categorize(res.cut)
  res.cat$Topic1 <- as.factor(res.cat$Topic1)
  res.cat$Topic2 <- as.factor(res.cat$Topic2)
  res.cat$Topic3 <- as.factor(res.cat$Topic3)
  res.cat$Topic4 <- as.factor(res.cat$Topic4)
  res.cat$Topic5 <- as.factor(res.cat$Topic5)
  res.cat$Topic6 <- as.factor(res.cat$Topic6)
  
  custom_theme <- function() {
    theme_classic(base_size = 10)+
      theme(
        plot.title=element_text(hjust=0.5, size = 10),
        axis.title.x = element_text(size = 10),  # Set X-axis title font size
        axis.title.y = element_text(size = 10),  # Set Y-axis title font size
        axis.text = element_text(size = 10),  # Set axis text font size
        legend.position = "none"
      )
  }
  
  sfit <- survfit(Surv(time, recurrence) ~ Topic1, data = res.cat)
  p1 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic1",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  sfit <- survfit(Surv(time, recurrence) ~ Topic2, data = res.cat)
  p2 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic2",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  sfit <- survfit(Surv(time, recurrence) ~ Topic3, data = res.cat)
  p3 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic3",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  sfit <- survfit(Surv(time, recurrence) ~ Topic4, data = res.cat)
  p4 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic4",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  sfit <- survfit(Surv(time, recurrence) ~ Topic5, data = res.cat)
  p5 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic5",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  sfit <- survfit(Surv(time, recurrence) ~ Topic6, data = res.cat)
  p6 = ggsurvplot(sfit, conf.int = TRUE, 
                  data = res.cat,
                  palette = c("dodgerblue2", "orchid2"), 
                  title = "Topic6",
                  legend.labs = c("Low", "High"), 
                  legend.title = "Group", 
                  ggtheme=custom_theme(),
                  ylim = c(0.3, 1))$plot
  plots = list(p1,p2,p3,p4,p5,p6)
  combined_plot <- ggarrange(plotlist = plots, 
                             ncol = 2, nrow = 3, 
                             common.legend = TRUE, 
                             legend = "bottom")
  print(combined_plot)
}



  

