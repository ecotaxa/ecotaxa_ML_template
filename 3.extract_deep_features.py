#!/usr/bin/python3
#
# Extract features from a trained feature extractor
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

# set a memory limit on tensorflow, to allow others to use the GPU too
gpus = tf.config.experimental.list_physical_devices('GPU')
# # either allow memory to grow as needed (less efficient -- and seems broken)
# tf.config.experimental.set_memory_growth(gpus[0], True)
# or set a predefined memory limit
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2300)])

batch_size = 16  # size of images batches in GPU memory
workers = 10     # number of parallel threads to prepare batches



print('Load feature extractor and dimensionality reducer') ## ----

my_fe = tf.keras.models.load_model('out/feature_extractor')
# get model input shape
input_shape = my_fe.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)

with open('out/dim_reducer.pickle','rb') as pca_file:
    pca = pickle.load(pca_file)


print('Load data and extract features') ## ----

for source in ['training', 'unknown']:
    # read DataFrame with image ids, paths and labels
    # NB: those would be in the database in EcoTaxa
    df = pd.read_csv('data/'+source+'_labels.csv', index_col='id')

    # prepare data batches
    batches = dataset.EcoTaxaGenerator(
        images_paths=df.img_path.values,
        input_shape=input_shape,
        labels=None, classes=None,
        # NB: we don't need the labels here, we just run images through the network
        batch_size=batch_size, augment=False, shuffle=False)

    # extract features by going through the batches
    full_features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)
    # and reduce their dimension
    reduced_features = pca.transform(full_features)
    
    # save them to disk
    reduced_features_df = pd.DataFrame(reduced_features, index=df.index)
    reduced_features_df.to_csv('data/'+source+'_deep_features.csv')
