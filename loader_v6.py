# import IPython.display as display

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import datetime, os
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score, f1_score
import config_v6 as cfg
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import models.Unet_v2 as nets
import models.Attention_R2_Unet_et_autres as attnet
import time
from IPython.display import clear_output
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence
import pathlib
import random
from PIL import Image
from glob import glob

# For more information about AUTOTUNE:
# https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"[INFO]...running Tensorflow ver. {tf.__version__}")
tf.config.run_functions_eagerly(False)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        print(e)

# list des images / masks pour TRAIN et VAL
glob_train_imgs = os.path.join(cfg.TRAIN_IMG , 'ID_*.tif')
glob_train_masks = os.path.join(cfg.TRAIN_MASK , 'ID_*.tif')

glob_val_imgs = os.path.join(cfg.VAL_IMG, 'ID_*.tif')
glob_val_masks = os.path.join(cfg.VAL_MASK, 'ID_*.tif')

glob_test_imgs = os.path.join(cfg.TEST_IMG, 'ID_*.tif')
glob_test_masks = os.path.join(cfg.TEST_MASK, 'ID_*.tif')

train_img_paths = glob(glob_train_imgs)
#train_mask_paths = glob(glob_train_masks)
val_img_paths = glob(glob_val_imgs)
#val_mvalpaths = glob(glob_val_masks)
test_img_paths = glob(glob_test_imgs)
#val_test_paths = glob(glob_test_masks)

def main():
    starttime = time.time()
    TRAINSET_SIZE = len(train_img_paths)
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    VALSET_SIZE = len(val_img_paths)
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")



    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()

    STEPS_PER_EPOCH = len(train_img_paths) // cfg.BATCH_SIZE
    VALIDATION_STEPS = len(val_img_paths) // cfg.BATCH_SIZE

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #loss = tf.keras.losses.CategoricalCrossentropy()


    # ----- tf.data TRAIN GENERATOR ----- #
    list_train_ds = tf.data.Dataset.from_tensor_slices(train_img_paths).shuffle(len(train_img_paths))
    print('len list', list_train_ds)


    train_dataset = list_train_ds.map(cfg.parse_train_image_all, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size=8000, seed=cfg.SEED)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(cfg.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


    print(train_dataset)

    # ----- tf.data VAL GENERATOR ----- #
    list_val_ds = tf.data.Dataset.from_tensor_slices(val_img_paths)

    val_dataset = list_val_ds.map(cfg.parse_val_image_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.BATCH_SIZE)

    print(val_dataset)

    # ----- tf.data TEST GENERATOR ----- #
    list_test_ds = tf.data.Dataset.from_tensor_slices(test_img_paths)

    test_dataset = list_test_ds.map(cfg.parse_val_image_all).take(20)
    test_dataset = test_dataset.batch(cfg.BATCH_SIZE)

    print(test_dataset)

    # ----- distribute dataset across GPUs (multi-worker ONLY) ----- #
    #dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    #dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    with strategy.scope():
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
        #class_weights = {0: 1, 1: 2, 3: 1, 4: 10, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1}
        #class_weight = [1, 4, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #loss = cfg.weighted_categorical_crossentropy(class_weight)
        loss = tf.keras.losses.CategoricalCrossentropy()

        callbacks = [
            # to show samples after each epoch
            # DisplayCallback(),
            # to collect some useful metrics and visualize them in tensorboard
            tensorboard_callback,
            ReduceLROnPlateau(factor=0.1, patience=6, min_lr=0.0000001, verbose=1),
            # if no accuracy improvements we can stop the training directly
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            # to save checkpoints
            tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True,
                                           save_weights_only=True)
        ]

        #optimizer = tfa.optimizers.RectifiedAdam(lr=1e-4)
        #optimizer = tfa.optimizers.Lookahead(optimizer)
        optimizer = Adam(lr=5e-5)
        if cfg.type_modele == 'UNET':

            model = nets.build_UNet(n_input_channels=4, BATCH_SIZE=None, num_output_classes=18, pad='same', input_dim=(256, 256),
                                   base_n_filters=64, do_dropout=True, pourc_do=0.3,
                                   reduction_poids=None)

        elif cfg.type_modele == 'R2UNET':

            model = nets.build_R2UNet(n_input_channels=3, BATCH_SIZE=None, num_output_classes=151, pad='same', input_dim=(128, 128),
                                   base_n_filters=32, do_dropout=False, pourc_do=0,
                                   reduction_poids=None)
        elif cfg.type_modele == 'ATT-R2UNET':

            model = attnet.build_attention_r2_unet(n_input_channels=4, num_output_classes=18, input_dim=(256, 256),
                                                   features=64, depth=4, data_format='channels_last',
                                                   last_activation='softmax')
        else:
            # for debugging ----------mirroredstrategy.scope()--------------
            model = nets.resnet_jouet()


        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(train_dataset, epochs=cfg.EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VALIDATION_STEPS,
                                  validation_data=val_dataset, callbacks=callbacks, verbose=1)
    print('[INFO]... elapse time', time.time() - starttime)

    score = model.evaluate(test_dataset,verbose=1)
    print("Resultats finals:")
    print("  Perte sur les donnees-test:\t\t\t{:.6f}".format(score[0]))
    print("  Precision sur les donnees-test:\t\t{:.2f} %".format(
        score[1] * 100))

    tf.keras.models.save_model(model, '/home/jp/PycharmProjects/load-data/SAVED_MODELS/land_use_unet_all_images.h5')

if __name__ == '__main__':
    main()
