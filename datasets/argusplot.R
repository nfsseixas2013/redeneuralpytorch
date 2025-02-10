setwd("/home/nilton/redeneuralpytorch/datasets/plots/")
library(ggplot2)
library(data.table)

training_avg = as.data.frame(fread('training_average.csv'))
p = ggplot(training_avg, aes(x = episodes, y = avg_reward, color = mode, group = mode)) +
  geom_line() +
  labs(
    title = "",
    x = "Episodes",
    y = "Avg Reward",
    color = ""
  ) +  theme_minimal() + 
  theme(legend.position = c(0.9, 0.2), legend.background = element_rect(fill="white", color="black"),
        text = element_text(size = 16, family = "Times New Roman"))

p
png("training_avg.png", units="in", width=10, height=5, res=300)
p
dev.off()

success_avg_training = as.data.frame(fread('success_average_training.csv'))
p = ggplot(success_avg_training, aes(x = episodes, y = `Success rate`, color = mode, group = mode)) +
  geom_line() +
  labs(
    title = "",
    x = "Episodes",
    y = "Avg Success",
    color = ""
  ) +   theme_minimal() + theme(legend.position = c(0.9, 0.2), legend.background = element_rect(fill="white", color="black"),
                                text = element_text(size = 16, family = "Times New Roman"))

png("training_succ.png", units="in", width=10, height=5, res=300)
p
dev.off()

inference_avg_test = as.data.frame(fread('inference_avg_reward_teste.csv'))
p = ggplot(inference_avg_test, aes(x = episodes, y = avg_reward, color = mode, group = mode)) +
  geom_line() +
  labs(
    title = "",
    x = "Episodes",
    y = "Avg Reward",
    color = ""
  ) +  theme_minimal() +theme(legend.position = c(0.9, 0.2), legend.background = element_rect(fill="white", color="black"),
                              text = element_text(size = 16, family = "Times New Roman"))
p
png("avg_reward_test.png", units="in", width=10, height=5, res=300)
p
dev.off()

inference_avg_training = as.data.frame(fread('inference_avg_reward_known_data.csv'))
p = ggplot(inference_avg_training, aes(x = episodes, y = avg_reward, color = mode, group = mode)) +
  geom_line() +
  labs(
    title = "",
    x = "Episodes",
    y = "Avg Reward",
    color = ""
  ) +  theme_minimal() + theme(legend.position = c(0.9, 0.2), legend.background = element_rect(fill="white", color="black"),
                               text = element_text(size = 16, family = "Times New Roman"))
p

png("avg_reward_training_data.png", units="in", width=10, height=5, res=300)
p
dev.off()

success_teste = as.data.frame(fread('inference_success_teste.csv'))
p = ggplot(success_teste, aes(x = episodes, y = `Success rate`, color = mode, group = mode)) +
  geom_line() +
  labs(
    title = "",
    x = "Episodes",
    y = "Avg Success",
    color = ""
  ) +  theme_minimal() + theme(legend.position = c(0.9, 0.2), legend.background = element_rect(fill="white", color="black"),
                               text = element_text(size = 16, family = "Times New Roman"))
p
png("avg_success_teste.png", units="in", width=10, height=5, res=300)
p
dev.off()

