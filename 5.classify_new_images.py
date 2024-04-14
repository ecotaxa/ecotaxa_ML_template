#!/usr/bin/env python
#
# CLassify images using a trained RandomForest classifier
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

import pickle

import pandas as pd
import numpy as np


print('Set options') ## ----

# None!


print('Read test images features') ## ----

hand_feat = pd.read_csv('io/test_features.csv.gz', index_col='id')
deep_feat = pd.read_csv('io/test_deep_features.csv.gz', index_col='id')

features = hand_feat.join(deep_feat)
# in EcoTaxa, this would be extracted from the database


print('Load and apply classifier') ## -----

with open('io/classifer.pickle','rb') as rf_file:
    RF = pickle.load(rf_file)
# in EcoTaxa, this is not necessary since the model will not be saved

probs = RF.predict_proba(features)

# get the list of classes, defined at the time the model is fitted
classes = RF.classes_

# extract highest score and corresponding label
predicted_scores = np.max(probs, axis=1)
predicted_labels = np.array(classes)[np.argmax(probs, axis=1)]


# compare with reality, just for fun
df = pd.read_csv('io/test_labels.csv', index_col='id')

from sklearn import metrics
metrics.accuracy_score(y_true=df.label, y_pred=predicted_labels)
metrics.log_loss(y_true=df.label, y_pred=probs)
