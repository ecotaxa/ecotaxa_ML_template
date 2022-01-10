#!/usr/bin/python3
#
# Train a deep network for plankton image classification
# NB: this step is done *outside* of EcoTaxa
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import tensorflow_tricks  # settings for tensorflow to behave nicely

import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics

from importlib import reload
import dataset            # custom data generator
import cnn                # custom functions for CNN generation
import biol_metrics       # custom functions model evaluation
dataset = reload(dataset)
cnn = reload(cnn)
biol_metrics = reload(biol_metrics)


print('Set options') ## ----

# I/O
# directory to save training checkpoints
ckpt_dir = 'io/checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)

# Data generator (see dataset.EcoTaxaGenerator)
batch_size = 16
augment = True
upscale = True
with open('io/crop.txt') as f:
    bottom_crop = int(f.read())

# CNN structure (see cnn.Create and cnn.Compile)
fe_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4'
input_shape = (224, 224, 3)
fe_trainable = True
# fc_layers_sizes = [1792, 896]
fc_layers_sizes = [600]
fc_layers_dropout = 0.4
classif_layer_dropout = 0.2

# CNN training (see cnn.Train)
use_class_weight = True
weight_sensitivity = 0.25  # 0.5 = sqrt
lr_method = 'decay'
initial_lr = 0.0005
decay_rate = 0.97
loss = 'cce'
epochs = 2
workers = 10


print('Prepare datasets') ## ----

# read DataFrame with image ids, paths and labels
# NB: those would be in the database in EcoTaxa
df = pd.read_csv('io/training_labels.csv', index_col='id')

# extract a validation set to monitor performance while training
seed = 1
# 75% in train
df_train = df.groupby('label').sample(frac=0.9, random_state=seed)
# the remaining 15% in val
df_val   = df.loc[set(df.index) - set(df_train.index)]

# count nb of examples per class in the training set
class_counts = df_train.groupby('label').size()
class_counts

# list classes
classes = class_counts.index.to_list()

# generate categories weights
# i.e. a dict with format { class number : class weight }
if use_class_weight:
    max_count = np.max(class_counts)
    class_weights = {}
    for idx,count in enumerate(class_counts.items()):
        class_weights.update({idx : (max_count / count[1])**weight_sensitivity})
else:
    class_weights = None

# define numnber of  classes to train on
nb_of_classes = len(classes)

# define data generators
train_batches = dataset.EcoTaxaGenerator(
    images_paths=df_train['img_path'].values,
    input_shape=input_shape,
    labels=df_train['label'].values, classes=classes,
    batch_size=batch_size, augment=augment, shuffle=True,
    crop=[0,0,bottom_crop,0])

val_batches = dataset.EcoTaxaGenerator(
    images_paths=df_val['img_path'].values,
    input_shape=input_shape,
    labels=df_val['label'].values, classes=classes,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])
# NB: do not suffle or augment data for validation, it is useless

total_batches = dataset.EcoTaxaGenerator(
    images_paths=df['img_path'].values,
    input_shape=input_shape,
    labels=None, classes=None,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])


print('Prepare model') ## ----

# try loading the model from a previous training checkpoint
my_cnn,initial_epoch = cnn.Load(ckpt_dir)

# if nothing is loaded this means the model was never trained
# in this case, define it
if (my_cnn is not None) :
    print('  restart from model trained until epoch ' + str(initial_epoch))
else :
    print('  define model')
    # define CNN
    my_cnn = cnn.Create(
        # feature extractor
        fe_url=fe_url,
        input_shape=input_shape,
        fe_trainable=fe_trainable,
        # fully connected layer(s)
        fc_layers_sizes=fc_layers_sizes,
        fc_layers_dropout=fc_layers_dropout,
        # classification layer
        classif_layer_size=nb_of_classes,
        classif_layer_dropout=classif_layer_dropout
    )

    print('  compile model')
    # compile CNN
    my_cnn = cnn.Compile(
        my_cnn,
        initial_lr=initial_lr,
        lr_method=lr_method,
        decay_steps=len(train_batches),
        decay_rate=decay_rate,
        loss=loss
    )

print('Train model') ## ----

# train CNN
history = cnn.Train(
    model=my_cnn,
    train_batches=train_batches,
    valid_batches=val_batches,
    epochs=epochs,
    initial_epoch=initial_epoch,
    class_weight=class_weight,
    log_frequency=1,
    class_weight=class_weights,
    output_dir=ckpt_dir,
    workers=workers
)


print('Evaluate model') ## ----

# load model for best epoch
best_epoch = None  # use None to get latest epoch
my_cnn,epoch = cnn.Load(ckpt_dir, epoch=best_epoch)
print(' at epoch {:d}'.format(epoch))

# predict classes for all dataset
pred = cnn.Predict(
    model=my_cnn,
    batches=total_batches,
    classes=classes,
    workers=workers
)

# comput a few scores, just for fun
df['predicted_label'] = pred
df.to_csv('io/predictions.csv')
# metrics.confusion_matrix(y_true=df.label, y_pred=df.predicted_label)
biol_metrics.classification_report(y_true=df.label, y_pred=df.predicted_label,
  non_biol_classes = ['badfocus<artefact', 'bubble', 'detritus', 'fiber<detritus'])


print('Save model') ## ----

# save model
my_cnn.save('io/cnn_model', include_optimizer=False)
# NB: do not include the optimizer state: (i) we don't need to retrain this final
#     model, (ii) it does not work with the native TF format anyhow.
