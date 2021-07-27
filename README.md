# EcoTaxa ML

The goal of this repository is to provide a functional example of the ML pipeline in EcoTaxa.

## Vocabulary

class = EcoTaxa category.  
label = name of an EcoTaxa category.  

training/fitting a model = tuning a model to adapt itself to the example data it is provided with.  
evaluating/predicting = using a trained model on a new set of data (to predict a class for each object, to compute features from it).  

learning/training set = set of objects with a label assigned (i.e. validated in EcoTaxa).  
validation set = separate set of objects, also labelled, used to check the performance of a model while training. 
test set = separate set of objects, also labelled, used to compute the final performance of a model after training.  
unlabelled set = set of objects with no labels, to be predicted.  

object features = numeric values that describe the object on the image: size, darkness, etc.  
handcrafted features = features that are extracted by a deterministic algorithm (before the upload in EcoTaxa).  
deep features = features that are extracted by a deep, convolutional neural network.  


## General process

1. for each object, people upload an image together with handcrafted features
2. EcoTaxa uses a pre-trained deep learning network (chosen per project) to extract more features for each object and stores them in the database
3. the user choses a learning set of objects
4. both handcrafted and deep features are extracted from the database and used to train a classifier
5. the user has a set of unlabelled images he/she wants to classify, for which we have the same handcrafted and deep features
6. the trained classifier is evaluated on this features table and predicts a class and a score for each image, which are stored in the database

NB: A current function of the application is the ability to save the trained classifier after step 4, to be able to use it without re-training. This use case has problems (https://github.com/ecotaxa/ecotaxa_dev/issues/151) and is dropped for now.


## Step 1: Train CNN feature extractor

This part is done *outside* of EcoTaxa.

A CNN has two parts:
1. the feature extractor, that does convolutions
2. the classifier, that takes the features as input and outputs the class "probabilities" (this is a Multi Layer Perceptron)
Both are trained at the same time

The process is:
- pick a feature extractor pre-trained on ImageNet
- add a few layers for classification
- re-train (i.e. fine tune) the CNN to classify a plankton dataset: this modifies the feature extractor slightly
- cut the classification layers out and save the rest
=> Get a feature extractor "best" adapted to the plankton data at hand.

For the training to work, images need to be input to the model at the appropriate scale, in batches, and possibly with slight modifications (i.e. data augmentation). This is the job of the data generator.

For the feature extractor:
Input: one image
Output: hundreds to thousands of features


## Step 2: Train a dimensionality reduction method

This part is done *outside* of EcoTaxa.

Because we cannot store thousands of numbers for each image in the database, the features need to be reduced to a smaller number. This is acceptable because many are redundant.

The features extracted for all images in the learning set are used to fit a Principal Component Analysis model, which computes new features that are linear correlation of the original features. These Principal Components are ordered (PC1 is the one of maximum variability among learning set images) and different from each other (i.e. orthogonal in the original multidimensional space).

The fitted object is saved and the same transformation from features to PC can be redone.

For the dimensionality reducer
Input: a vector of hundreds to thousands of features
Output: a chosen number (currently 50) of new features, ordered in decreasing order of importance


## Step 3: Extract deep features

This is done in EcoTaxa, currently the first time that the user wants to use the deep features of a set of images. Could also be done right after import when the deep feature extractor has been chosen.

The process is:
- load the feature extractor
- feed images to the feature extractor through the data generator and generate features
- load the dimensionality reducer
- feed the features to the dimensionality reducer
- store the resulting, reduced and ordered features

The feature extractor + dimensionality reducer are a "network" (currently called SCN network). The old ones are not compatible anymore. We will produce new ones based on the work of Thelma.


## Step 4: Train classifier

This is done in EcoTaxa, every time a user wants to make a prediction.

The process is:
- load the handcrafted features, deep features, and labels for the training set
- train the classifier (a RandomForest)
- optionally, store the classifier on disk (this is the part that will be removed because of issue #151)

For the classifier
Input: a vector of features
Output: a vector of probabilities to be in each class


## Step 5: Classify new images

This is done in EcoTaxa, most of the time right after step 4 (from the point of view of the user, steps 4 and 5 are a single step).

The process is:
- load the handcrafted and deep features for the dataset to predict
- load the classifier
- evaluate the classifier on the features
- find the class of maximum probability per image
- store the max proba and corresponding label in the database



