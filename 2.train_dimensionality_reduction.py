#!/usr/bin/python3
#
# Train a PCA to reduce the number of features produced by a deep feature extractor
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import os
# disable tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # does not work on the command line...
# store models downloaded from TFHub in the user's home to avoid permission problems
os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.tfhub_modules/')
import pickle

import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

import dataset   # custom data generator


print('Set options') ## ----

batch_size = 16  # size of images batches in GPU memory
workers = 10     # number of parallel threads to prepare batches
n_dims = 50      # number of dimensions to keep after dimensionality reduction



print('Load model and create feature extractor') ## ----

# load saved model
my_cnn = tf.keras.models.load_model('out/best_model')

# get model input shape
input_shape = my_cnn.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)

# drop the Dense and Dropout layers to get only the feature extractor
my_fe = tf.keras.models.Sequential(
    [layer for layer in my_cnn.layers
     if not (isinstance(layer, tf.keras.layers.Dense) |
             isinstance(layer, tf.keras.layers.Dropout))
    ])
my_fe.summary()

# save feature extractor
my_fe.save('out/feature_extractor')


print('Load data and extract features for the training set') ## ----

# read DataFrame with image ids, paths and labels
# NB: those would be in the database in EcoTaxa
df = pd.read_csv('data/training_labels.csv', index_col='id')

# prepare data batches
batches = dataset.EcoTaxaGenerator(
    images_paths=df.img_path.values,
    input_shape=input_shape,
    labels=None, classes=None, 
    batch_size=batch_size, augment=False, shuffle=False)

# extract features by going through the batches
features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)


print('Define dimensionality reduction') ## ----

# define the PCA
pca = PCA(n_components=n_dims)
# fit it to the training data
pca.fit(features)

# save it for later application
with open('out/dim_reducer.pickle', 'wb') as pca_file:
    pickle.dump(pca, pca_file)
