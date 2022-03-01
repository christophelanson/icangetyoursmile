import numpy as np
import PIL
from PIL import Image
import os
from tensorflow.keras.utils import image_dataset_from_directory
import random

def get_dataset(path, sample_size=500, image_size=(64,64), random_seed=1):
    """
    get a dataset of images of required size, randomly selected
    returns X (masked images), y (unmasked images of the same faces), and a sample test set of 5 images
    path to data : ..../raw_data
    the function then completes the path by adding : 64x64/Mask or No_mask (or 256x256/Mask etc.)
    """
    path = f'{path}/{image_size[0]}x{image_size[1]}/'
    print(f'Loading data from {path}...')
    random.seed(random_seed)
    data_size = sample_size
    X = []
    y = []
    X_test = []
    for number in range(data_size):
        rand_img_nb = random.randint(0,9999)
        no_mask_path = f'{path}No_mask/seed{str(rand_img_nb).zfill(4)}.png'
        no_mask_im = np.asarray(Image.open(no_mask_path)).tolist()
        mask_path = f'{path}Mask/with-mask-default-mask-seed{str(rand_img_nb).zfill(4)}.png'
        mask_im = np.asarray(Image.open(mask_path)).tolist()
        X.append(mask_im)
        y.append(no_mask_im)
    for number in range(5):
        rand_img_nb = random.randint(0,9999)
        mask_path = f'{path}Mask/with-mask-default-mask-seed{str(rand_img_nb).zfill(4)}.png'
        mask_im = np.asarray(Image.open(mask_path)).tolist()
        X_test.append(mask_im)
    print('Done')
    print(f'X shape : {np.asarray(X).shape}')
    print(f'y shape : {np.asarray(y).shape}')
    print(f'X_test shape : {np.asarray(X_test).shape}')
    return X, y, X_test

def plot_results(X_test, y_pred):
    """
    plots masked graphs and their predicted faces
    """
    plt.figure(figsize=(20,10))
    nb_graphs = len(X_test)
    for graph_nb in range(nb_graphs):
        plt.subplot(2,nb_graphs, graph_nb+1)
        plt.imshow(X_test[graph_nb])
        plt.subplot(2,nb_graphs, graph_nb +1 +nb_graphs)
        plt.imshow(y_pred[graph_nb])





### this is an olkd function to retrieve a batch dataset that we may use later

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
