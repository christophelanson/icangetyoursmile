from optparse import Values
from google.cloud import storage

import pandas as pd

import numpy as np
import glob

import os
from icangetyoursmile.utils import run_full_model
from tensorflow.keras.models import load_model

#from dotenv import dotenv_values
#settings = dotenv_values() # dictionnary of settings in .env file
# environment variables defined here
JOB_NAME='icgys_model_traing'
BUCKET_NAME='i-can-get-your-smile'
BUCKET_PACKAGE_FOLDER='package_folder'
BUCKET_STORAGE_FOLDER='storage'
PACKAGE_NAME='icangetyoursmile'
FILENAME='trainer'
PYTHON_VERSION='3.7'
RUNTIME_VERSION='2.8'
REGION='europe-west1'
BUCKET_STORAGE_FOLDER='storage'
PROJECT_ID='le-wagon-337814'


# calculated environment variables
MODEL_NAME = 'full-Unet-model'
MODEL_VERSION = 'img10k-epoch2000-pwr3-gcp'


def upload_model_to_gcp(model_name, run_locally=True):

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    if run_locally==True:
        # image log
        print('uploading image_log to gcp')
        blob = bucket.blob(f'{BUCKET_STORAGE_FOLDER}/{model_name}.pickle')
        blob.upload_from_filename(f'./image_logs/{model_name}_img_log.pickle')
    # model
    print('looking for files to upload')
    current_wd = os.getcwd()
    for root, directories, files in os.walk('./saved_models'):
        for name in files:
            full_name = os.path.join(root.replace(current_wd,""), name)
            if model_name in full_name: # otherwise will upload ALL saved models
                print('uploading :',full_name)
                blob = bucket.blob(f'{BUCKET_STORAGE_FOLDER}/{full_name.strip("./saved_models/")}')
                blob.upload_from_filename(f'{full_name}')
    print('upload finished')
    print('all done')


if __name__ == '__main__':
    model_name = f'{MODEL_NAME}-{MODEL_VERSION}'
    run_locally = False
    path_to_data = None # for notebook use None, else gcp path to data (below)
    if run_locally == True:
        #local parameters
        unet_power=3
        sample_size=20
        epochs=2
        image_size=(64,64)
        batch_size=8
    if run_locally == False :
        path_to_data = f'https://console.cloud.google.com/storage/browser/{BUCKET_NAME}'
        # gcp ai model parameters
        unet_power=5
        sample_size=10000
        epochs=2000
        image_size=(64,64)
        batch_size=32
    run_full_model(model_name, run_locally=run_locally, unet_power=unet_power, sample_size=sample_size,
                   epochs=epochs, image_size=image_size, random_seed=2,
                   test_split=0.15, batch_size=batch_size, validation_split=0.2)
    upload_model_to_gcp(model_name, run_locally=run_locally)


def download_model_from_gcp(model_name="full-Unet-model"):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(f"{BUCKET_NAME}")

    folder = model_name
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder + "/variables" )

    blobs = bucket.list_blobs(prefix=f"storage/{model_name}/")
    for i, blob in enumerate(blobs):
        blob = blob
        blob.download_to_filename(folder + "/" + blob.name.replace(f'storage/{model_name}/',""))
    model = load_model(folder)
    return model
