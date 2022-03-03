import numpy as np
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


def create_dataset(path_to_images, image_size=(64,64), batch_size=32, validation_split=0.2, random_seed=1):
    path_to_images += f'{image_size[0]}x{image_size[1]}'

    """ Create train and validation datasets from images with a classification smile/no_smile.
    """

    train_dataset = image_dataset_from_directory(
        directory=path_to_images,
        labels="inferred",
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="training",
        seed=random_seed)

    validation_dataset = image_dataset_from_directory(
        directory=path_to_images,
        labels="inferred",
        batch_size=batch_size,
        validation_split=validation_split,
        image_size=image_size,
        subset="validation",
        seed=random_seed)


    return train_dataset, validation_dataset


def Convo2D(filters, kernel_size=4,padding="same", activation="LeakyReLU"):

    """
    Create Conv2D Layers to invoke in the initialize model
    """
    first = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          padding=padding,
                          activation=activation)
    second = layers.Conv2D(filters=filters,
                          kernel_size=2,
                          padding=padding,
                          activation=activation)
    third = layers.MaxPooling2D()

    return [first, second, third]

def initialize_model(input_shape=(64,64,3)):

    model = Sequential(
    [
        layers.InputLayer(input_shape)
    ]
    + Convo2D(8) + Convo2D(16) + Convo2D(32) + Convo2D(64) +
    [layers.Flatten(),
     layers.Dropout(0.35),
     layers.Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.01)),
     layers.Dense(1, activation="sigmoid"),
     ])

    model.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy"])

    return model



# Get the absolute path to feed the create_dataset function

#absolute_path = os.path.dirname(os.path.dirname(os.getcwd()))
#path = absolute_path + "/raw_data/sns/"

# Get class names from the dataset

#class_names = train_dataset.class_names
#class_names

# Plot the 9 first images from the dataset with classification

#plt.figure(figsize=(10, 10))
#for images, labels in train_dataset.take(1):
  #for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
