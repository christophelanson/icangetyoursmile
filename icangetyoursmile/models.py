import numpy as np
import os
import skimage.io as io
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as kera
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

#Unet is the "masque" models

def unet(starting_power = 3,pretrained_weights = None,input_size = (64,64,3)):
    inputs = Input(input_size)
    factor = 2 ** starting_power
    conv1 = Conv2D(factor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(factor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(factor * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(factor * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(factor * 2 * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(factor * 2 * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(factor * (2 ** 3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(factor * (2 ** 3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(factor * (2 ** 4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(factor * (2 ** 4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(factor * (2 ** 3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(factor * (2 ** 3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(factor * (2 ** 3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(factor * (2 * 2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(factor * (2 * 2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(factor * (2 * 2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(factor * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(factor * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(factor * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(factor, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(factor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(factor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)


 #   conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
 #   conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(3, 1, activation = 'linear')(conv9)


    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'mse')

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def create_data_augmentation_model(random_flip="horizontal", random_rotation=0.03):
    """
    creates a model of data augmentation using Sequential
    hyperparameters :
    random_flip = "horizontal" "vertical" "horizontal and vertical"
    random_rotation = 0.03 (0.03 == 3% x 360??)
    """
    model = Sequential([
            layers.RandomFlip(random_flip),
            layers.RandomRotation(random_rotation),
            ])
    return model


def save_model(model, model_name):
    """
    Save the model in saved_model folder.
    Enter a model and a model name as a string
    """
    model.save(f'../../saved_model/{model_name}')


def join_unet_augm_models(unet_model, augmentation_model):
    model = Sequential([
          augmentation_model,
          unet_model
          ])
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'mse')
    return model
