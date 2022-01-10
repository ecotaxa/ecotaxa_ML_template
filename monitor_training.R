#
# Plot training progress
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")
library("patchwork")

# read training log
input <- "io/checkpoints/training_log.tsv"
d <- read_tsv(input, col_types = cols()) %>%
  # reformat to get losses and accuracies in the same column
  pivot_longer(cols=c(starts_with("train"), starts_with("val"))) %>%
  separate(name, into=c("dataset", "metric"), sep="_") %>%
  pivot_wider(names_from=metric, values_from=value)

# define plots
p_l <- ggplot(d) + geom_path(aes(x=step, y=loss, colour=dataset)) + theme(legend.position="top")
p_a <- ggplot(d) + geom_path(aes(x=step, y=accuracy, colour=dataset)) + theme(legend.position="top")
p_lr <- ggplot(filter(d, dataset=="train")) + geom_path(aes(x=step, y=learning_rate))

# plot in a nice grid
(p_l + p_a) / p_lr + plot_layout(heights=c(3,1))
