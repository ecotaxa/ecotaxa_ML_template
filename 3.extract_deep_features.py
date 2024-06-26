#!/usr/bin/env python
#
# Extract features from a trained feature extractor
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import tensorflow_tricks  # settings for tensorflow to behave nicely

import pickle

import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

import dataset   # custom data generator


print('Set options') ## ----

batch_size = 16  # size of images batches in GPU memory
workers = 10     # number of parallel threads to prepare batches
with open('io/crop.txt') as f:
    bottom_crop = int(f.read())

print('Load feature extractor and dimensionality reducer') ## ----

my_fe = tf.keras.models.load_model('io/feature_extractor', compile=False)
# get model input shape
input_shape = my_fe.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)

with open('io/dim_reducer.pickle','rb') as pca_file:
    pca = pickle.load(pca_file)


print('Load data and extract features') ## ----

for source in ['training', 'validation', 'test']:
    # read DataFrame with image ids, paths and labels
    # NB: those would be in the database in EcoTaxa
    df = pd.read_csv('io/'+source+'_labels.csv', index_col='id')

    # prepare data batches
    batches = dataset.EcoTaxaGenerator(
        images_paths=df.img_path.values,
        input_shape=input_shape,
        labels=None, classes=None,
        # NB: we don't need the labels here, we just run images through the network
        batch_size=batch_size, augment=False, shuffle=False,
        crop=[0,0,bottom_crop,0])

    # extract features by going through the batches
    full_features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)
    # and reduce their dimension
    reduced_features = pca.transform(full_features)

    # save them to disk
    reduced_features_df = pd.DataFrame(reduced_features, index=df.index)
    reduced_features_df.to_csv('io/'+source+'_deep_features.csv.gz', compression="gzip")
