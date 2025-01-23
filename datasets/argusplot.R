setwd("/home/nilton/redeneuralpytorch/datasets/")
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
  )

png("training_avg.png", units="in", width=8, height=5, res=300)
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
  )
png("training_succ.png", units="in", width=8, height=5, res=300)
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
  )
p
png("avg_reward_test.png", units="in", width=8, height=5, res=300)
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
  )
p

png("avg_reward_training_data.png", units="in", width=8, height=5, res=300)
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
  )
p
png("avg_success_teste.png", units="in", width=8, height=5, res=300)
p
dev.off()
