import numpy as np
import PIL
from PIL import Image
import os
import pickle
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model
import random
from icangetyoursmile.models import *
from icangetyoursmile.custom_callbacks import CustomCallback
from tensorflow.keras.callbacks import EarlyStopping

def get_dataset_tts(path, sample_size=500, image_size=(64,64), random_seed=1, test_split=0.15):
    """
    get a dataset of images of required size, randomly selected
    returns X (masked images), y (unmasked images of the same faces), and a sample test set of 5 images
    path to data : ..../raw_data
    the function then completes the path by adding : 64x64/Mask or No_mask (or 256x256/Mask etc.)
    """
    path = f'{path}/{image_size[0]}x{image_size[1]}/'
    print(f'Loading data from {path}...')
    random.seed(random_seed)
    X = []
    y = []
    X_test = []
    y_test = []
    X_visu = []
    y_visu = []
    photo_numbers = random.sample(list(range(10000)), sample_size)
    test_size = int(sample_size * test_split)
    for number in photo_numbers[0:sample_size-test_size]:
        no_mask_path = f'{path}No_mask/seed{str(number).zfill(4)}.png'
        no_mask_im = np.asarray(Image.open(no_mask_path)).tolist()
        mask_path = f'{path}Mask/with-mask-default-mask-seed{str(number).zfill(4)}.png'
        mask_im = np.asarray(Image.open(mask_path)).tolist()
        X.append(mask_im)
        y.append(no_mask_im)
    for number in photo_numbers[sample_size-test_size:sample_size]:
        no_mask_path = f'{path}No_mask/seed{str(number).zfill(4)}.png'
        no_mask_im = np.asarray(Image.open(no_mask_path)).tolist()
        mask_path = f'{path}Mask/with-mask-default-mask-seed{str(number).zfill(4)}.png'
        mask_im = np.asarray(Image.open(mask_path)).tolist()
        X_test.append(mask_im)
        y_test.append(no_mask_im)
    for number in random.sample(photo_numbers[sample_size-test_size:sample_size],5):
        no_mask_path = f'{path}No_mask/seed{str(number).zfill(4)}.png'
        no_mask_im = np.asarray(Image.open(no_mask_path)).tolist()
        mask_path = f'{path}Mask/with-mask-default-mask-seed{str(number).zfill(4)}.png'
        mask_im = np.asarray(Image.open(mask_path)).tolist()
        X_visu.append(mask_im)
        y_visu.append(no_mask_im)


    print('Done')
    print(f'X shape : {np.asarray(X).shape}')
    print(f'y shape : {np.asarray(y).shape}')
    print(f'X_test shape : {np.asarray(X_test).shape}')
    print(f'y_test shape : {np.asarray(y_test).shape}')
    print(f'X_visu shape : {np.asarray(X_visu).shape}')
    print(f'y_visu shape : {np.asarray(y_visu).shape}')
    return np.array(X), np.array(y), np.array(X_test), np.array(y_test), np.array(X_visu), np.array(y_visu)

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


def plot_loss(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS 1 ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    #ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Test'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- LOSS 2 ---

    starting_epoch = int(len(history.history['loss']) * 0.66)
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylim((0,history.history['loss'][starting_epoch]))
    ax[1].set_xlim((starting_epoch,len(history.history['loss']) -1))
    ax[1].legend(['Train', 'Test'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)


def save_model(model, model_name):
    """
    Save the model in saved_models folder.
    Enter a model and a model name as a string
    """
    model.save(f'./saved_models/{model_name}')


def loading_model(model_name):
    """
    Load the model in saved_model folder.
    """
    return load_model(f'./saved_models/{model_name}')

def run_full_model(define_model_name, path_to_data=None, run_locally=True,unet_power=3, sample_size=500, epochs=50, image_size=(64,64), random_seed=1, test_split=0.15, batch_size=8, validation_split=0.2):
    if run_locally == True:
        absolute_path = '/home/christophelanson/code/christophelanson/icangetyoursmile'
        #os.path.dirname(os.path.dirname(os.getcwd()))
        path_to_data = absolute_path + "/raw_data"

    X, y, X_test, y_test, X_visu, y_visu = get_dataset_tts(path_to_data, sample_size=sample_size, image_size=image_size, random_seed=random_seed, test_split=test_split)

    input_size = (image_size[0], image_size[1],3)
    model = join_unet_augm_models(unet(starting_power=unet_power, input_size=input_size),create_data_augmentation_model())

    X_visu_image_log = dict() #log of model predict(X_visu) for each epoch to animate fit results
    callback_save_X_visu_predict = CustomCallback(X_visu, X_visu_image_log)
    early_stopping = EarlyStopping(patience= 40, restore_best_weights=True)

    results = model.fit(X, y, batch_size=batch_size, epochs=epochs, use_multiprocessing=True,
                        validation_split=validation_split,
                        callbacks = [callback_save_X_visu_predict, early_stopping]
                        )
    save_model(model, define_model_name)
    with open(f'./image_logs/image_log{define_model_name}.pickle', 'wb') as handle:
        pickle.dump(X_visu_image_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if run_locally == True:
        y_pred_visu = model.predict(X_visu).astype(np.uint8)
        plot_results(X_visu, y_pred_visu)
        plot_loss(results)
        score = model.evaluate(X_test, y_test)
        return f"Model {define_model_name} saved, mse-score: {score}"


def plot_results(X_visu, y_pred):
    """
    plot 2 lines of 5 graphs
    first line shows X_visu (ie X test)
    second line shows y_pred from X_visu
    """
    plt.figure(figsize=(20,10))
    nb_graphs = len(X_visu)
    for graph_nb in range(nb_graphs):
        plt.subplot(2,nb_graphs, graph_nb+1)
        plt.imshow(X_visu[graph_nb])
        plt.subplot(2,nb_graphs, graph_nb +1 +nb_graphs)
        plt.imshow(y_pred[graph_nb])

def animate_results(X_visu_image_log, image_nb):
    """
    animates one of X_visu images (choosen by image_nb) from X_visu_image_log
    """
    fig = plt.figure(figsize=(3,3))
    frames = []
    for i in range(len(X_visu_image_log)):
        frames.append([plt.imshow(X_visu_image_log[i][1],animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=1000)
    plt.show()

    # To save the animation, use e.g.
    # ani.save("movie.mp4")
    # or
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
