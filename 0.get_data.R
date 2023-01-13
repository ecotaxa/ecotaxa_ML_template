#
# Get some test data from Thelma's data sets
#

library("tidyverse")

# read everything
d <- read_csv("/home/tpanaiotis/Documents/These/pcb_datasets/out/zooscan/zooscan_data.csv") %>%
  rename(label=classif_id_2)

# determine dominant classes
dominant <- count(d, label) %>%
  arrange(n) %>%
  tail(4)

# keep 100 examples per class, randomly
set.seed(1)
subset <- d %>%
  filter(label %in% dominant$label) %>%
  group_by(label) %>% sample_n(100, replace=FALSE) %>%
  mutate(set=c(rep("training", 80), rep("test", 20))) %>%
  ungroup() %>%
  mutate(
    id=1:n(),
    img_path=str_c("data/images/", id, ".jpg")
  )

# write files to disc
subset %>%
  group_by(set) %>%
  do({
    write_csv(select(., id, img_path, label), str_c("data/", .$set[1], "_labels.csv"))
    write_csv(select(., id, area:cdexc), str_c("data/", .$set[1], "_features.csv"))
  })

# copy image files
file.copy(str_c("/home/tpanaiotis/Documents/These/pcb_datasets/out/zooscan/", subset$path_to_img), subset$img_path)

