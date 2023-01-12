#
# Plot training progress
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")
library("patchwork")

io <- "io/"

# read training log
d <- read_tsv(str_c(io, "/checkpoints/training_log.tsv"), col_types = cols())
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


## Plot training stats ----

source("lib_ml_utils.R")
p <- read_csv(str_c(io, "/cnn_predictions.csv"), col_types=cols()) %>% rename(true=label, pred=predicted_label)
plot(cm(p$true, p$pred))
ggsave(str_c(io, "/cnn_confusion_matrix.png"), width=8, height=8)
crp <- plot(classification_report(p$true, p$pred, exclude=c("artefact", "crystal", "detritus", "fiber", "other<living")))
crp
gtsave(crp, str_c(io, "/cnn_classification_report.html"))

