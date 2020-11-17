import numpy as np
import os
import tensorflow as tf
from glob import glob
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.layers import *
import tensorflow_io as tfio
from tensorflow.keras.utils import to_categorical

SEED = 42
EPOCHS = 20
type_modele = 'ATT-R2UNET'
#type_modele = 'UNET'
K.set_image_data_format('channels_last')
# image format == tif
# dataset_path = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/'
# training_data = 'train_frames/img/'
# val_data = 'val_frames/img/'
# Image size that we are going to use
IMG_SIZE = (256, 256)
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 18
BATCH_SIZE = 12
BATCH_SIZE_REPLICA = BATCH_SIZE / 2


# MODIFY THIS SECTION TO REFLECT IMAGE LOCATION IN YOUR ENVIRONMENT
TRAIN_IMG = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/'
# train mask directory
TRAIN_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/'
# train mask directory
TRAIN_MASK = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/'
# validation images directory
TRAIN_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/'
# validation images directory
VAL_IMG = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/val_frames/img/'
# validation mask directory
VAL_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/val_dsm/img/'
# validation mask directory
VAL_MASK = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/val_masks/img/'
# saved model directory
VAL_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/val_osm/img/'
# saved model directory
TEST_IMG = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/test_frames/img/'
# validation mask directory
TEST_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/test_dsm/img/'
# validation mask directory
TEST_MASK = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/test_masks/img/'
# saved model directory
WEIGHT_PATH = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/models/'
# log dir for tensorboard
T_BOARD = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/tensorboard/'
# you can change this path to reflect your own settings if necessary
# dataset_path = "/media/DATA/DATA_LANDUSE/ADEchallenge/ADEChallengeData2016/images/"
# dataset_path = "/home/jp/dataset/ADEChallengeData2016/images/"
# training_data = "training/"
# val_data = "validation/"




def parse_image2(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    tensor
    """
    # -----------------------RGB--------------------- #
    # LOAD RGB IMAGES IN FOLDER
    image = tf.io.read_file(img_path)
    image = tfio.experimental.image.decode_tiff(image)
    # image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_crop(image, size=(256, 256, 3), seed=SEED)
    # image = tf.expand_dims(image, axis=0)

    # For RGB Image path, derive ALL PATH with Regex:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/img/ID_001.tif'
    # Its corresponding mask path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/img/ID_001.tif'
    # Its corresponding dsm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/img/ID_001.tif'
    # Its corresponding osm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/img/ID_001.tif'

    # -----------------------MASK--------------------- #
    # MASK path
    mask_path = tf.strings.regex_replace(img_path, "train_frames", "train_masks")
    # mask_path = tf.strings.regex_replace(mask_path, "tif", "png")
    # print('..........................mask path',tf.print(mask_path))
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask = tf.image.random_crop(mask, size=(256, 256, 1), seed=SEED)
    # mask = tf.expand_dims(mask, axis=0)
    mask = tf.one_hot(tf.squeeze(mask, axis=2), N_CLASSES)

    return image, mask


def parse_image3(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    tensor
    """
    local_seed = tf.random.uniform([SEED])
    # -----------------------RGB--------------------- #
    # LOAD RGB IMAGES IN FOLDER
    image = tf.io.read_file(img_path)
    image = tfio.experimental.image.decode_tiff(image)
    # image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_crop(image, size=(256, 256, 3), seed=local_seed)
    # image = tf.expand_dims(image, axis=0)

    # For RGB Image path, derive ALL PATH with Regex:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/img/ID_001.tif'
    # Its corresponding mask path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/img/ID_001.tif'
    # Its corresponding dsm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/img/ID_001.tif'
    # Its corresponding osm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/img/ID_001.tif'

    # -----------------------MASK--------------------- #
    # MASK path
    mask_path = tf.strings.regex_replace(img_path, "val_frames", "val_masks")
    # mask_path = tf.strings.regex_replace(mask_path, "tif", "png")
    # print('..........................mask path',tf.print(mask_path))
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask = tf.image.random_crop(mask, size=(256, 256, 1), seed=local_seed)
    # mask = tf.expand_dims(mask, axis=0)
    mask = tf.one_hot(tf.squeeze(mask, axis=2), N_CLASSES)

    return image, mask

@tf.function
def parse_train_image_all(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    tensor

    # SEED config : https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    """
    tf.random.uniform([SEED])
    # -----------------------RGB--------------------- #
    # LOAD RGB IMAGES IN FOLDER
    image = tf.io.read_file(img_path)

    image = tfio.experimental.image.decode_tiff(image)
    # image = tf.keras.preprocessing.image.load_img(str(img_path))
    # image = tf.keras.preprocessing.image.array_to_img(image)
    # image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    #image = tf.image.random_crop(image, size=(256, 256, 3), seed=SEED)
    # image = tf.expand_dims(image, axis=0)

    # For RGB Image path, derive ALL PATH with Regex:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/img/ID_001.tif'
    # Its corresponding mask path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/img/ID_001.tif'
    # Its corresponding dsm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/img/ID_001.tif'
    # Its corresponding osm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/img/ID_001.tif'

    # -----------------------MASK--------------------- #
    # MASK path
    mask_path = tf.strings.regex_replace(img_path, "train_frames", "train_masks")
    # mask_path = tf.strings.regex_replace(mask_path, "tif", "png")
    # print('..........................mask path',tf.print(mask_path))
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    #mask = tf.image.random_crop(mask, size=(256, 256, 1), seed=SEED)
    # mask = tf.expand_dims(mask, axis=0)
    #mask = tf.one_hot(tf.squeeze(mask, axis=2), N_CLASSES)

    # -----------------------DSM--------------------- #
    # DSM path


    dsm_path = tf.strings.regex_replace(img_path, "train_frames", "train_dsm")
    dsm = tf.io.read_file(dsm_path)
    # The masks contain a class index for each pixels
    dsm = tfio.experimental.image.decode_tiff(dsm)
    dsm = tf.cast(dsm, tf.float32) / 255.
    #dsm = tf.cast(dsm, tf.float32) # samples already saved in [0,1] interval
    #dsm = tf.image.random_crop(dsm, size=(256, 256, 1), seed=SEED)

    # -----------------------OSM--------------------- #
    # OSM path


    osm_path = tf.strings.regex_replace(img_path, "train_frames", "train_osm")
    osm = tf.io.read_file(osm_path)
    # The masks contain a class index for each pixels
    osm = tfio.experimental.image.decode_tiff(osm)
    osm = tf.cast(osm, tf.float32) * 14 / 255.  #  normalize around 1

    #osm = tf.image.random_crop(osm, size=(256, 256, 1), seed=SEED)

    # generater Tensor (256, 256, 5)
    #final = tf.concat([image, dsm, osm], axis=2)
    final = tf.concat([image, dsm, mask], axis=2)
    print('tenseur',final)

    final_out  = tf.image.random_crop(final, size=[256, 256, 5])

    #final_out = tf.image.random_crop(final, size=(256, 256, 23))
    img_out, mask_out = tf.split(final_out, [4,1],axis=2)
    mask_out = tf.cast(mask_out, tf.uint8)
    #mask_out = tf.cast(mask_out, tf.uint8)

    return img_out, tf.one_hot(tf.squeeze(mask_out, axis=2), N_CLASSES)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap

@tf.function
def parse_val_image_all(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    tensor
    """
    #tf.random.uniform([SEED])
    # -----------------------RGB--------------------- #
    # LOAD RGB IMAGES IN FOLDER
    image = tf.io.read_file(img_path)
    image = tfio.experimental.image.decode_tiff(image)
    # image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    #image = tf.image.random_crop(image, size=(256, 256, 3), seed=SEED)
    # image = tf.expand_dims(image, axis=0)

    # For RGB Image path, derive ALL PATH with Regex:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/img/ID_001.tif'
    # Its corresponding mask path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/img/ID_001.tif'
    # Its corresponding dsm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/img/ID_001.tif'
    # Its corresponding osm path is:
    # .../DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/img/ID_001.tif'

    # -----------------------MASK--------------------- #
    # MASK path
    mask_path = tf.strings.regex_replace(img_path, "val_frames", "val_masks")
    # mask_path = tf.strings.regex_replace(mask_path, "tif", "png")
    # print('..........................mask path',tf.print(mask_path))
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    #mask = tf.image.random_crop(mask, size=(256, 256, 1), seed=SEED)
    # mask = tf.expand_dims(mask, axis=0)
    #mask = tf.one_hot(tf.squeeze(mask, axis=2), N_CLASSES)

    # -----------------------DSM--------------------- #
    # DSM path
    dsm_path = tf.strings.regex_replace(img_path, "val_frames", "val_osm")
    # dsm_path = tf.strings.regex_replace(mask_path, "tif", "png")
    # print('..........................mask path',tf.print(mask_path))
    dsm = tf.io.read_file(dsm_path)
    dsm = tfio.experimental.image.decode_tiff(dsm)
    dsm = tf.cast(dsm, tf.float32) / 255.
    # min_val = tf.math.reduce_min(dsm, axis=None, keepdims=True)
    # dsm = tf.math.subtract(dsm,min_val)
    #dsm = tf.image.random_crop(dsm, size=(256, 256, 1), seed=SEED)

    # -----------------------OSM--------------------- #
    # OSM path
    osm_path = tf.strings.regex_replace(img_path, "val_frames", "val_osm")
    osm = tf.io.read_file(osm_path)
    # The masks contain a class index for each pixels
    osm = tfio.experimental.image.decode_tiff(osm)
    osm = tf.cast(osm, tf.float32) * 14 / 255.  # normalize values 0-17 to 0-252, close enough

    #osm = tf.image.random_crop(osm, size=(256, 256, 1), seed=SEED)

    # generater Tensor (256, 256, 5)
    final = tf.concat([image, dsm,mask], axis=2)
    #print('tenseur',final)

    final_out  = tf.image.random_crop(final, size=[256, 256, 5])

    #final_out = tf.image.random_crop(final, size=(256, 256, 23))
    img_out, mask_out = tf.split(final_out, [4,1],axis=2)
    mask_out = tf.cast(mask_out, tf.uint8)
    #mask_out = tf.cast(mask_out, tf.uint8)

    return img_out, tf.one_hot(tf.squeeze(mask_out, axis=2), N_CLASSES)


# Decorator @tf.function
# if you want to know more about it:
# https://www.tensorflow.org/api_docs/python/tf/function
@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.one_hot(tf.squeeze(input_mask, axis=2), N_CLASSES)

    return input_image, input_mask


def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    # lb_x = np.random.randint(0, 1024 - IMG_SIZE[0])
    # lb_y = np.random.randint(0, 1024 - IMG_SIZE[1])

    input_image = tf.image.random_crop(datapoint['image'], size=(256, 256, 3), seed=SEED)
    input_mask = tf.image.random_crop(datapoint['segmentation_mask'], size=(256, 256), seed=SEED)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.random_crop(datapoint['image'], size=(256, 256, 3), seed=SEED)
    input_mask = tf.image.random_crop(datapoint['segmentation_mask'], size=(256, 256), seed=SEED)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

