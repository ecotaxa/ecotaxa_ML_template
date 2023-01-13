#
# Plot training progress
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")
library("patchwork")

# read training log
input <- "io/checkpoints/training_log.tsv"
d <- read_tsv(input, col_types = cols())
dm <- d %>%
  # reformat to get losses and accuracies in the same column
  pivot_longer(cols=c(starts_with("train"), starts_with("val"))) %>%
  separate(name, into=c("dataset", "metric"), sep="_") %>%
  pivot_wider(names_from=metric, values_from=value)
de <- filter(d, batch == max(batch))

# define plots
base <- ggplot() + geom_vline(aes(xintercept=step), data=de, colour="grey80") + theme(legend.position="top")
p_loss <- base + geom_path(aes(x=step, y=loss, colour=dataset), data=dm)
p_accu <- base + geom_path(aes(x=step, y=accuracy, colour=dataset), data=dm)
p_lr <- base + geom_path(aes(x=step, y=learning_rate), data=d)

# plot in a nice grid
(p_loss + p_accu) / p_lr + plot_layout(heights=c(2.5,1))
