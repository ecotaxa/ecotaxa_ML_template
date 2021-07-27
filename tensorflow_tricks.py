#
# Some settings to force TensorFlow to behave nicely
# 
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3


import os

# disable tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# NB: does not work when run on the command line apparently...

# store models downloaded from TFHub in the user's home to avoid permission problems
os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.tfhub_modules/')

# set a memory limit on tensorflow, which otherwise takes all the GPU memory by default
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

# # either allow memory to grow as needed (less efficient -- and seems broken)
# tf.config.experimental.set_memory_growth(gpus[0], True)

# or set a predefined memory limit
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2300)])

