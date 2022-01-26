#!/usr/bin/python3
#
# Train a PCA to reduce the number of features produced by a deep feature extractor
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import tensorflow_tricks  # settings for tensorflow to behave nicely

import pickle

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA

import dataset   # custom data generator


print('Set options') ## ----

batch_size = 16  # size of images batches in GPU memory
workers = 10     # number of parallel threads to prepare batches
n_dims = 50      # number of dimensions to keep after dimensionality reduction
with open('io/crop.txt') as f:  # number of pixels to crop at the bottom
    bottom_crop = int(f.read())

print('Load feature extractor') ## ----

my_fe = tf.keras.models.load_model('io/feature_extractor', compile=False)

# get model input shape
input_shape = my_fe.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)


print('Load data and extract features for the training set') ## ----

# read DataFrame with image ids, paths and labels
# NB: those would be in the database in EcoTaxa
df = pd.read_csv('io/training_labels.csv', index_col='id')

# prepare data batches
batches = dataset.EcoTaxaGenerator(
    images_paths=df.img_path.values,
    input_shape=input_shape,
    labels=None, classes=None,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])

# extract features by going through the batches
features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)


print('Fit dimensionality reduction') ## ----

# define the PCA
pca = PCA(n_components=n_dims)
# fit it to the training data
pca.fit(features)

# save it for later application
with open('io/dim_reducer.pickle', 'wb') as pca_file:
    pickle.dump(pca, pca_file)
