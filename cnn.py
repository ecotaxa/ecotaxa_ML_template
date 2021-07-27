#
# Functions to create, compile, train, and predict a CNN model
#
# (c) 2021 Thelma Panaiotis, Jean-Olivier Irisson, GNU General Public License v3

import os

import numpy as np

import tensorflow as tf
from tensorflow.keras import utils, layers, optimizers, losses, callbacks 
import tensorflow_hub as hub
import tensorflow_addons as tfa


def Create(
    fe_url, input_shape, fe_trainable,
    fc_layers_sizes, fc_layers_dropout,
    classif_layer_size, classif_layer_dropout
):

    """
    Generates a CNN model. 
    
    Args:
        fe_url (str): URL of the feature extractor on TF Hub
        input_shape (list, int): dimensions of the input image in the network
                                 (property of the feature extractor)
        fe_trainable (bool): whether to train the feature extractor (True) or
                             only the fc + classification layers (False)
        fc_layers_sizes (list of int): size of fully connected layers 
        fc_layers_dropout (float): dropout of fully connected layers 
        classif_layer_size (int): size of classification layer
                                  (i.e. number of classes)
        classif_layer_dropout (float): dropout of classification layer

    Returns:
        model (tf.keras.Sequential): CNN model
    """
    
    # Initiate empty model
    model = tf.keras.Sequential()
    
    # Get feature extractor from TF hub
    fe_layer = hub.KerasLayer(fe_url, input_shape=input_shape)
    # set feature extractor trainability
    fe_layer.trainable = fe_trainable
    model.add(fe_layer)
    
    # Add fully connected layers
    for i in range(len(fc_layers_sizes)):
        if fc_layers_dropout:
            model.add(layers.Dropout(fc_layers_dropout))
        model.add(layers.Dense(fc_layers_sizes[i], activation='relu'))
    
    # Add classification layer
    if classif_layer_dropout:
        model.add(layers.Dropout(classif_layer_dropout))
    model.add(layers.Dense(classif_layer_size, activation='softmax'))

    # print model summary
    model.summary()

    return model


def Compile(
    model, initial_lr, lr_method='constant',
    decay_steps=1.0, decay_rate=0.5, loss='cce'
):
    """
    Compiles a CNN model. 
    
    Args:
        model (tf.keras.Sequential): CNN model to compile
        initial_lr (float): initial learning rate. If `lr_method`='constant', this is the learning rate.
        lr_method (str): method for learning rate.
            'constant' for a constant learning rate
            'decay' for a learning rate decaying with time
        decay_steps (int): number of optimiser steps (i.e. data batches) used to compute the decay of
            the learning rate.
        decay_rate (float): rate of learning rate decay. The actual decay is computed as:
                initial_lr / (1 + decay_rate * step / decay_steps)
            where step is one optimiser step (i.e. one data batch).
        loss (str): loss function.
          'cce' for CategoricalCrossentropy
          (see https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy),
          'sfce' for SigmoidFocalCrossEntropy
          (see https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy),
          useful for unbalanced classes
    
    Returns:
        model (tf.keras.Sequential): compiled CNN model
        
    """
    # Define learning rate
    if lr_method == 'decay':
        lr = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    else: # Keep constant learning rate
        lr = initial_lr
    
    # Define optimizer
    optimizer = optimizers.Adam(learning_rate=lr)
    
    # Define loss
    if loss == 'cce':
        loss = losses.CategoricalCrossentropy(from_logits=True,
                   reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
        # TODO consider using
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
        # to avoid having to one-hot encode the labels
    elif loss == 'sfce':
        loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,
                   reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    # Compile model
    model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=['accuracy']
    )
    
    return model


def Load(output_dir='.'):
    """
    Load a CNN model.
    
    Args:
        output_dir (str): path to the directory where the model has been saved
    
    Returns:
        model (tf.keras.Sequential): CNN model
        epoch (int): latest training epoch for the loaded model,
            to start re-training at
    """
    # list existing model training checkpoints
    checkpoints = os.listdir(output_dir)
    # TODO check difference with tf.saved_model.load()
    
    if len(checkpoints) > 0 :
        checkpoints.sort(reverse=True)
        latest_checkpoint = os.path.join(output_dir, checkpoints[0])

        # load the model
        model = tf.keras.models.load_model(latest_checkpoint)
        model.summary()
        
        # get epoch from file name
        epoch = int(latest_checkpoint.split('.')[1])
        # TODO: check if there is a more robust way to get this from the model
    else :
        model = None
        epoch = 0
    
    return model,epoch


def Train(
    model, train_batches, valid_batches,
    epochs, initial_epoch=0, class_weight=None, output_dir='.', workers=1
):
    """
    Trains a CNN model. 
    
    Args:
        model (tf.keras.Sequential): CNN model to train
        train_batches (dataset.EcoTaxaGenerator): batches of data for training
        valid_batches (dataset.EcoTaxaGenerator): batches of data for validation
        epochs (int): number of epochs to train for
        class_weight (dict): weights for classes
        output_dir (str): directory where to save model weights

    Returns:
        history (tf.keras.callbacks.History) that contains loss and accuracy for
        the traning and validation dataset.
    """
    
    # Set callbacks to save model weights when validation loss improves
    checkpoint_path = os.path.join(output_dir, "checkpoint.{epoch:03d}.ckpt")
    # NB: hdf5 does not work for saving full models
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        save_freq='epoch',
        verbose=1
    )
    
    # Fit the model
    history = model.fit(
        x=train_batches,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        initial_epoch=initial_epoch,
        validation_data=valid_batches,
        class_weight=class_weight,
        max_queue_size=max(10, workers*2),
        workers=workers
    )
        
    return history


def Predict(model, batches, classes=None, workers=1):
    """
    Predict batches from a CNN model
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
        batches (dataset.EcoTaxaGenerator): batches of data to predict
        classes (list): None or list of class names; when None, the function
            returns the content of the classification layer 
        workers (int): number of CPU workers to prepare data
    
    Returns:
        prediction (ndarray): with as many rows as input and, as columns:
            logits when `classes` is None
            class names when `classes` is given
        
    """
    
    # Predict all batches
    prediction = model.predict(
        batches,
        max_queue_size=max(10, workers*2),
        workers=workers
    )
    # NB: pred is an array with:
    # - as many lines as there are items in the batches to predict
    # - as many columns as the size of the output layer of the model
    # and it contains the models' output
    
    if classes is not None:
        # convert it to predicted classes
        prediction = np.array(classes)[np.argmax(prediction, axis=1)]

    return prediction
