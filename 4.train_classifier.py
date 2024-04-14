#!/usr/bin/env python
#
# Train a RandomForest classifier
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


print('Set options') ## ----

n_estimators = 300        # number of trees in the RF
min_samples_leaf = 5      # min number of object per leaf
class_weight = 'balanced' # class weights inversely proportional to class count
n_jobs = 10               # number of parallel threads


print('Read training labels and features') ## ----

labels    = pd.read_csv('io/training_labels.csv', index_col='id')
hand_feat = pd.read_csv('io/training_features.csv.gz', index_col='id')
deep_feat = pd.read_csv('io/training_deep_features.csv.gz', index_col='id')

# combine handcrafted and deep features
features = hand_feat.join(deep_feat)

print('Define and train classifier') ## -----

RF = RandomForestClassifier(n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            class_weight=class_weight,
                            n_jobs=n_jobs)

RF.fit(X=features, y=labels.label.values)

# save it to disk for the next step
with open('io/classifer.pickle','wb') as rf_file:
    pickle.dump(RF, rf_file)
# NB: In EcoTaxa, this used to be possible but resulted in very large models, which were nearly unusable.
#     So this feature is removed for now and the prediction (step 5) is done right after the fitting of the model.
