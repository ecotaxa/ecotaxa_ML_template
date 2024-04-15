#!/usr/bin/env python
#
# Read the training log and plot the current stats
#
# (c) 2023 Jean-Olivier Irisson, GNU General Public License v3

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('io/checkpoints/training_log.tsv', sep='\t')
df = df.drop(['batch', 'learning_rate'], axis='columns')

df.plot(x='step', subplots=True)
plt.show()

df.plot(x='epoch', subplots=True)
plt.show()
