import numpy as np
import PIL
from PIL import Image
import os
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

## Absolute path to images ##

#absolute_path = os.path.dirname(os.path.dirname(os.getcwd()))
#path = absolute_path + "/raw_data/"

## Fonction to create train and validation datasets for X (No_mask images) and y(Mask images), return 4 datasets

def create_train_val_dataset(path_to_images,
                             image_size=(64,64),
                             batch_size=32,
                             validation_split=0.2,
                             random_seed=1):
    """
    Create from directory of images one train and one validation dataset for Mask images(X) and No_mask images(y)

    """

    path_to_images += f'{image_size[0]}x{image_size[1]}/'

    # Path to folder Mask
    path_to_images_mask = path_to_images + "Mask"

    # Create train and validation datasets for X

    X_train_dataset = image_dataset_from_directory(
        directory=path_to_images_mask,
        labels=None,
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="training",
        seed=random_seed,
    )

    X_val_dataset = image_dataset_from_directory(
        directory=path_to_images_mask,
        labels=None,
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="validation",
        seed=random_seed,
    )

    # Path to folder No_mask
    path_to_images_no_mask = path_to_images + "No_mask"

    # Create train and validation datasets for y

    y_train_dataset = image_dataset_from_directory(
        directory=path_to_images_no_mask,
        labels=None,
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="training",
        seed=random_seed,
    )

    y_val_dataset = image_dataset_from_directory(
        directory=path_to_images_no_mask,
        labels=None,
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="validation",
        seed=random_seed,
    )

    return X_train_dataset, X_val_dataset, y_train_dataset, y_val_dataset


def plot_loss(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Test'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)
