#!/usr/bin/python3
#
# Train a deep network for plankton image classification
# NB: this step is done *outside* of EcoTaxa
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import tensorflow_tricks  # settings for tensorflow to behave nicely

import os
import tensorflow as tf

import cnn                # custom functions for CNN generation

# read model weights
my_cnn = tf.keras.models.load_model('out/best_model')

# drop the Dense and Dropout layers to get only the feature extractor
my_fe = tf.keras.models.Sequential(
    [layer for layer in my_cnn.layers
     if not (isinstance(layer, tf.keras.layers.Dense) |
             isinstance(layer, tf.keras.layers.Dropout))
    ])
my_fe.summary()

# save feature extractor
my_fe.save('out/feature_extractor')
# or the following, for a single file archive
my_fe.save('out/feature_extractor.hdf5')
