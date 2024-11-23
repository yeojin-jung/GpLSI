library(ggplot2)
library(reshape2)
library(tidyverse)
library(gridExtra)
library(dplyr)
library(vroom)
library(alto)
library(pROC)
library(compositions)
library(RColorBrewer)

source(file.path(getwd(), '/utils/alto.R'))
source(file.path(getwd(), '/utils/roc.R'))

##### Simulation #####
model_root = file.path(getwd(), 'simulation/results_final')
file_path = file.path(model_root, 'results_all.csv')

##### Stanford-crc dataset #####
model_root = file.path(getwd(), '/data/stanford-crc')

## 1. plot topic resolution
## a. alto
# 400 x 300
ntopics = 6
model_path = '/Users/jeong-yeojin/Dropbox/SpLSI/data/stanford-crc/model/model_3hop'
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
line_color = hcl.pals()(n = 3, name = "Set3")
line_color = c("#E78AC3", "#8DA0CB", "#A6D854")

crc_K_summ = crc_K_results %>%
  #filter(method == 'lda') %>%
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
p1 = ggplot(data = crc_K_summ, aes(x=K, y=med1, color=Method)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = med1-sd1, ymax = med1+sd1), width = 0.2) +
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
  )

# Plot for Cosine Similarity
p2 = ggplot(data = crc_K_summ, aes(x=K, y=med2, color=Method)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = med2-sd2, ymax = med2+sd2), width = 0.2) +
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
  )

# Arrange plots side by side
grid.arrange(p1, p2, ncol=2)


## 2. survival analysis
## a. plot roc (topic proportion, thresholding) (1 x 2)
data_files = c('survival_gplsi.csv', 'survival_plsi.csv', 'survival_lda.csv')
data_paths = c()
for(i in 1:length(data_files)){
  data_paths[i] = file.path(model_path, data_files[i])
}
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
results_spleen <- read_csv("~/Dropbox/SpLSI/data/spleen/model/results_spleen.csv")
results_spleen$method = factor(results_spleen$method, levels = c('GpLSI', 'pLSI', 'TopicScore','LDA', 'SLDA'))
p1 = ggplot(data = results_spleen,
            aes(x=K, y=moran, group=method)) +
  geom_line(aes(color=method)) +
  geom_point(aes(color=method)) +
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
  )

p2 = ggplot(data = results_spleen,
            aes(x=K, y=pas, group=method)) +
  geom_line(aes(color=method)) +
  geom_point(aes(color=method)) +
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
  )

grid.arrange(p1, p2, ncol=1)


##### WHat's cooking dataset #####
## 1, 

