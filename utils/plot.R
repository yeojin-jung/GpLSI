library(ggplot2)
library(reshape2)
library(tidyverse)
library(gridExtra)
library(dplyr)
library(tidytext)
library(vroom)
library(alto)
library(pROC)
library(compositions)
library(RColorBrewer)

source(file.path(getwd(), '/utils/alto.R'))
source(file.path(getwd(), '/utils/roc.R'))


##### Stanford-crc dataset #####
model_root = file.path(getwd(), 'data/stanford-crc')

## 1. plot topic resolution
## a. alto
# 400 x 300
ntopics = 6
model_path = file.path(model_root, 'model/model_3hop')
immune_names = c('CD4 T cell', 'CD8 T cell', 'B cell', 'Macrophage', 'Granulocyte', 'Blood vessel', 'Stroma', 'Other')
methods = c('gplsi', 'plsi', 'lda')
# 500 x 350
for(method in methods){
  root_path_Ahat = paste0(model_path,'/Ahats_aligned/Ahats_',method)
  root_path_What = paste0(model_path,'/Whats_aligned/Whats_',method)
  p = get_paths(ntopics, immune_names, root_path_Ahat, root_path_What)
}

## b. batch splitting (1 x 3)
file_path = file.path(model_path, 'crc_chooseK_results.csv')
crc_K_results <- read_csv(file_path)

crc_K_summ = crc_K_results %>%
  mutate(Method = factor(method, levels = c("gplsi", "plsi", "lda"), labels = c("GpLSI", "pLSI", "LDA"))) %>%
  group_by(K, Method) %>%
  summarize(
    med1 = median(l1_dist, na.rm = TRUE),
    sd1 = IQR(l1_dist, na.rm = TRUE)/2,
    med2 = median(cos_sim, na.rm = TRUE),
    sd2 = IQR(cos_sim, na.rm = TRUE)/2,
    med3 = median(cos_sim_ratio, na.rm = TRUE),
    sd3 = IQR(cos_sim_ratio, na.rm = TRUE)/2,
  )

# Plot for L1 Distance
# 280 x 300
p1 = ggplot(data = crc_K_summ, aes(x=K, y=med1, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med1-sd1, ymax = med1+sd1, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "L1 Distance",
  ) +
  #ylim(0, 0.5) +
  scale_y_log10()+
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Plot for Cosine Similarity
p2 = ggplot(data = crc_K_summ, aes(x=K, y=med2, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med2-sd2, ymax = med2+sd2, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "Cosine Similarity",
  ) +
  scale_y_log10()+
  #ylim(0.7, 1.0) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Plot for Cosine Similarity ratio
p3 = ggplot(data = crc_K_summ, aes(x=K, y=med3, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med3-sd3, ymax = med3+sd3, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "Cosine Similarity Ratio",
  ) +
  scale_y_log10()+
  #ylim(0.7, 1.0) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
    
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Arrange plots side by side
grid.arrange(p1, p2, ncol=2)
grid.arrange(p1, p2, p3, ncol=3)

## 2. survival analysis
## a. plot roc (topic proportion, thresholding) (1 x 2)
# 430 x 300
func = process_data_binary
func = process_data_prop

auc_results <- data.frame(K = integer(), AUC = numeric(), Method = character())
for(ntopic in 2:6){
  data_files = c(paste0('survival_gplsi_K=',ntopic,'.csv'),
                 paste0('survival_plsi_K=',ntopic,'.csv'),
                 paste0('survival_lda_K=',ntopic,'.csv'))
  data_paths = c()
  for(i in 1:length(data_files)){
    data_paths[i] = file.path(model_path, data_files[i])
  }
  method_names = c('GpLSI', 'pLSI', 'LDA')
  results = lapply(data_paths, func, nruns=10)
  for (i in seq_along(results)) {
    auc_values <- results[[i]]$auc_values
    auc_results <- rbind(
      auc_results,
      data.frame(K = ntopic, AUC = auc_values, Method = method_names[i])
    )
  }
}

auc_summ = auc_results %>%
  mutate(Method = factor(Method, levels = c("GpLSI", "pLSI", "LDA"), labels = c("GpLSI", "pLSI", "LDA"))) %>%
  group_by(K, Method) %>%
  summarize(
    med = median(AUC, na.rm = TRUE),
    sd = IQR(AUC, na.rm = TRUE)/2,
  )
p2 = ggplot(auc_summ, aes(x = K, y = med, group = Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_line(aes(color=Method)) + 
  #geom_errorbar(aes(ymin = med-sd, ymax = med+sd), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "AUC",
    color = "Method"
  ) +
  ylim(0.5, 0.7) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(size = 14),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.position = "right"
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38","#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7)) # Custom shapes

grid.arrange(p1, p2, ncol=2)
# 280 x 280
nruns=10
roc_data = plot_roc(process_data_binary, data_paths, 'binary', nruns=nruns)
roc_data = plot_roc(process_data_prop, data_paths, nruns=nruns)

## b. (optional) plot coef (topic proportion, dichotomized) (1 x 2) and KM plot
p_gplsi = plot_coefs(data_paths[1], ntopics)
p_plsi = plot_coefs(data_paths[2], ntopics)
p_lda = plot_coefs(data_paths[3], ntopics)

plots <- list(p_gplsi, p_plsi, p_lda)
combined_plot <- ggarrange(plotlist = plots, 
                           ncol = 3, 
                           common.legend = TRUE, 
                           legend = "bottom",
                           align = "h") # Align horizontally with added space
print(combined_plot)

# 600 x 620
plot_KM(data_paths[1])
plot_KM(data_paths[2])
plot_KM(data_paths[3])



##### Spleen Mouse dataset #####
## 1. plot smoothness scores (1 x 3)
## a. moran's I, pas, time
model_root = file.path(getwd(), 'data/spleen')
model_path = file.path(model_root, 'model')
file_path = file.path(model_path, 'results_spleen.csv')
results_spleen= read_csv(file_path)
results_spleen$method = factor(results_spleen$method, levels = c('GpLSI', 'pLSI', 'TopicScore','LDA', 'SLDA'))
p1 = ggplot(data = results_spleen,
            aes(x=K, y=moran, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  labs(
    title = NULL,
    x = "K",
    y = "Moran's I",
  ) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  )+
  scale_x_log10(breaks = unique(results_spleen$K))+
  scale_color_manual(values = c("#F8766D","#00BA38","#00BFC4","#619CFF","#F564E3")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 3, 7, 8)) # Custom shapes
  

p2 = ggplot(data = results_spleen,
            aes(x=K, y=pas, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  labs(
    title = NULL,
    x = "K",
    y = "PAS",
  ) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  ) +
  scale_x_log10(breaks = unique(results_spleen$K))+
  scale_color_manual(values = c("#F8766D","#00BA38","#00BFC4","#619CFF","#F564E3")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 3, 7, 8)) # Custom shapes

grid.arrange(p1, p2, ncol=2)


##### WHat's cooking dataset #####
## 1, choose K
model_root = file.path(getwd(), 'data/whats-cooking')
model_path = file.path(model_root, 'model_final')

file_path = file.path(model_path, 'cook_chooseK_results.csv')
cook_K_results <- read_csv(file_path)

cook_K_summ = cook_K_results %>%
  mutate(Method = factor(method, levels = c("gplsi", "plsi", "lda"), labels = c("GpLSI", "pLSI", "LDA"))) %>%
  group_by(K, Method) %>%
  summarize(
    med1 = median(l1_dist, na.rm = TRUE),
    sd1 = IQR(l1_dist, na.rm = TRUE)/2,
    med2 = median(cos_sim, na.rm = TRUE),
    sd2 = IQR(cos_sim, na.rm = TRUE)/2,
    med3 = median(cos_sim_ratio, na.rm = TRUE),
    sd3 = IQR(cos_sim_ratio, na.rm = TRUE)/2,
  )

# Plot for L1 Distance
# 280 x 300
p1 = ggplot(data = cook_K_summ, aes(x=K, y=med1, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med1-sd1, ymax = med1+sd1, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "L1 Distance",
  ) +
  #ylim(0, 0.5) +
  scale_y_log10()+
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Plot for Cosine Similarity
p2 = ggplot(data = cook_K_summ, aes(x=K, y=med2, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med2-sd2, ymax = med2+sd2, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "Cosine Similarity",
  ) +
  scale_y_log10()+
  #ylim(0.7, 1.0) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Plot for Cosine Similarity ratio
p3 = ggplot(data = cook_K_summ, aes(x=K, y=med3, group=Method)) +
  geom_line(aes(color=Method)) +
  geom_point(size = 2, aes(shape=Method, color=Method)) +
  geom_errorbar(aes(ymin = med3-sd3, ymax = med3+sd3, color=Method), width = 0.2) +
  labs(
    title = NULL,
    x = "K",
    y = "Cosine Similarity Ratio",
  ) +
  scale_y_log10()+
  #ylim(0.7, 1.0) +
  theme_classic(base_size = 10)+
  theme(
    plot.title=element_text(hjust=0.5, size = 10),
    axis.title.x = element_text(size = 10),  # Set X-axis title font size
    axis.title.y = element_text(size = 10),  # Set Y-axis title font size
    axis.text = element_text(size = 10),  # Set axis text font size
    
  )+
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF")) +  # Custom colors
  scale_shape_manual(values = c(16, 15, 7))

# Arrange plots side by side
grid.arrange(p1, p2, p3, ncol=3)


## 2. plot topics
model_root = file.path(getwd(), 'data/whats-cooking')
model_path = file.path(model_root, 'model_final')

file_path = file.path(model_path, 'top_words_lda_7.csv')
anchor = read_csv(file_path)
colnames(anchor) = c('topic', 'anchor_words', 'anchor_weights')
anchor_sorted <- anchor %>%
  arrange(topic, desc(anchor_weights)) %>%
  mutate(anchor_words = reorder_within(anchor_words, anchor_weights, topic))

topic_labeller <- as_labeller(function(topic) paste("Topic", topic))

ggplot(anchor_sorted, aes(x = anchor_words, 
                          y = anchor_weights, 
                          fill = as.factor(topic))) +
  geom_bar(stat = "identity") +
  coord_flip() +
  facet_wrap(~topic, scales = "free", labeller = topic_labeller) +
  scale_x_reordered() + 
  labs(x = NULL, y = NULL, fill = "Topic") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10), 
    axis.text.x = element_text(size = 10), 
    strip.text = element_text(size = 12),  
    legend.position = "none" 
  )

