#
# Data generator for EcoTaxa images
#
# (c) 2021 Thelma Panaiotis, Jean-Olivier Irisson, GNU General Public License v3

import os

import numpy as np

import cv2                            # image loading, manipulation
from imgaug import augmenters as iaa  # image manipulation

from sklearn.preprocessing import MultiLabelBinarizer # one-hot encoder

from tensorflow.keras import utils

class EcoTaxaGenerator(utils.Sequence):
    """
    Generates data batches

    Args:
        images (list/ndarray, str): paths to images
        input_shape (tuple, int): dimensions of the input images for the network
        labels (list/ndarray, str): image labels (as text) or None for an unlabelled batch
        classes (list, str): all classes to consider for the classification,
            or None for an unlabelled batch
        batch_size (int): number of images in a batch
            larger means faster/better training/evaluation but higher GPU memory use
        shuffle (bool) : whether to shuffle inputs between epochs
            usually True for training, False otherwise
        augment (bool) : whether to use data augmentation on the input
            usually True for training, False otherwise
        upscale (bool) : whether to scale small images up to the input shape
            if False, small images are just padded
        crop (tuple, int) : number of pixels to crop from the top,right,bottom,left.

    Returns:
        A batch of `batch_size` images (4D ndarray) and one-hot encoded labels (2D ndarray)

    """
    def __init__(self, images_paths, input_shape, labels=None, classes=None, batch_size=32,
                 shuffle=False, augment=False, upscale=True, crop=(0,0,0,0)):
        'Initialization of settings'
        # initialize constants
        self.images_paths = images_paths
        self.labels       = labels
        self.input_shape  = input_shape
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.augment      = augment
        self.upscale      = upscale
        self.crop         = crop

        if (labels is not None):
            # initialise the one-hot encoder
            mlb = MultiLabelBinarizer(classes=classes)
            self.class_encoder = mlb

        # print('Nb of images : ' + str(len(self.images_paths)))
        # print('Nb of labels : ' + str(len(self.labels)))
        # print('Input shape : ' + str(self.input_shape))
        # print('Batch size : ' + str(self.batch_size))
        # print('Shuffle inputs : ' + str(self.shuffle))
        # print('Augment inputs : ' + str(self.augment))
        # print('Upscale small images : ' + str(self.upscale))

        self.on_epoch_end()

    def __len__(self):
        'Compute the number of batches to cover the dataset in one epoch'
        return int(np.ceil(len(self.images_paths) / self.batch_size))
        # NB: use np.ceil instead of np.floor to be sure to see all items
        #     (important for the test set in particular)

    def on_epoch_end(self):
        'Update indexes after each epoch'
        # reinitialise indexes
        self.indexes = np.arange(len(self.images_paths))
        # and, if chosen, shuffle them between epochs, to make sure the batches
        # are not always the same
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def padding_value(self, img):
        'Compute value to use to pad an image, as the median value of border pixels'
        # get height and width of image
        h,w = img.shape[0:2]

        # concatenate border pixels in an array
        borders = np.concatenate((
            img[:, 0],         # left column
            img[:, w-1],       # right column
            img[0, 1:w-2],     # top line without corners
            img[h-1, 1:w-2],   # bottom line without corners
        ), axis=0)

        # compute the median
        pad_value = np.median(borders)

        return pad_value

    def augmenter(self, images):
        """
        Define a data augmenter which does horizontalf flip (50% chance),
        vertical flip (50% chance), zoom and shear
        """
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    # TODO what happens when images are zoomed and there is not enough padding?
                    shear=(-15, 15),  # shear by -15 to +15 degrees
                    mode='edge', # pad images with border picels
                ),
            ],
            random_order=True # apply these transformations in random order
        )
        return seq(images=images)

    def __getitem__(self, index):
        'Generate one batch of data'

        # pick indexes of images for this batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # select and load images from this batch
        batch_paths = [self.images_paths[i] for i in indexes]
        batch_orig_images = [cv2.imread(p)/255. for p in batch_paths]

        # resize images to the input dimension of the network
        batch_prepared_images = []
        input_size = self.input_shape[0] # NB: assumes square input
        for img in batch_orig_images:
            # crop a defined number of pixels from the sides
            [t,r,b,l] = self.crop
            h,w = img.shape[0:2]
            img = img[t:(h-b),l:(w-r),:]
            # TODO crop to object, automatically

            h,w = img.shape[0:2]

            # compute largest dimension (hor or ver)
            dim_max = max(h,w)

            # upscale small images or downscale large ones
            if (self.upscale) or (dim_max > input_size):
                # resize image so that largest dim is now equal to input_size
                img = cv2.resize(
                    img,
                    (max(h*input_size//dim_max,1), max(w*input_size//dim_max,1)),
                    interpolation=cv2.INTER_AREA
                )
                h,w = img.shape[0:2]

            # TODO review the following for speed, possibly

            # create a square, empty output, of desired dimension, filled with padding value
            pad_value = self.padding_value(img)
            img_square = np.full(self.input_shape, pad_value)

            # compute number of pixels to leave blank
            offset_ver = int((input_size-h)/2) # on top and bottom of image
            offset_hor = int((input_size-w)/2) # on left and right of image

            # replace pixels by input image
            img_square[offset_ver:offset_ver+h, offset_hor:offset_hor+w] = img
            batch_prepared_images.append(img_square)

        # convert to array of images
        batch_prepared_images = np.array([img for img in batch_prepared_images], dtype='float32')

        # augment images
        if self.augment == True:
            batch_prepared_images = self.augmenter(batch_prepared_images)

        # extract the labels corresponding to the selected indexes, when they are provided
        if self.labels is not None:
            batch_labels = [self.labels[i] for i in indexes]
            batch_encoded_labels = self.class_encoder.fit_transform([[l] for l in batch_labels])
        else :
            batch_encoded_labels = None

        # return reshaped images and labels
        return batch_prepared_images,batch_encoded_labels

