import numpy as np
import PIL
from PIL import Image
import os
from tensorflow.keras.utils import image_dataset_from_directory


## Absolute path to images ##

absolute_path = os.path.dirname(os.path.dirname(os.getcwd()))
path = absolute_path + "/raw_data/"

## Fonction to create train and validation datasets for X (No_mask images) and y(Mask images), return 4 datasets

def create_train_val_dataset(path_to_image,
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
