from optparse import Values
from google.cloud import storage
import pandas as pd
import numpy as np
import glob
from google.cloud import storage
import os
from icangetyoursmile.utils import run_full_model

from dotenv import dotenv_values
settings = dotenv_values() # dictionnary of settings in .env file
# get environment variables
BUCKET_NAME=settings['BUCKET_NAME']
BUCKET_PACKAGE_FOLDER=settings['BUCKET_PACKAGE_FOLDER']
BUCKET_STORAGE_FOLDER=settings['BUCKET_STORAGE_FOLDER']
PACKAGE_NAME=settings['PACKAGE_NAME']
FILENAME=settings['FILENAME']
PYTHON_VERSION=settings['PYTHON_VERSION']
RUNTIME_VERSION=settings['RUNTIME_VERSION']
REGION=settings['REGION']

# environment variables defined here
JOB_NAME='icgys_model_traing'

# calculated environment variables
STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'
BUCKET_TRAIN_DATA_PATH='to be filled'
MODEL_NAME = 'full-Unet-model'
MODEL_VERSION = 'v1'

def upload_model_to_gcp(run_locally=True):

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # image log
    print('uploading image_log to gcp')
    blob = bucket.blob(f'{BUCKET_STORAGE_FOLDER}/{MODEL_NAME}.pickle')
    blob.upload_from_filename(f'./image_logs/{MODEL_NAME}.pickle')
    print('uploading model to gcp')
    current_wd = os.getcwd()
    for root, directories, files in os.walk('./saved_models'):
        for name in files:
            full_name = os.path.join(root, name).strip(current_wd)
            print('uploading :',full_name)
            blob = bucket.blob(f'{BUCKET_STORAGE_FOLDER}/{full_name.strip("./saved_models/")}')
            blob.upload_from_filename(f'{full_name}')



if __name__ == '__main__':
    print(FILENAME)
    model_name = MODEL_NAME
    run_locally = True
    path_to_data = None # for notebook use None, else gcp path to data
    run_full_model(model_name, run_locally=run_locally, path_to_data=path_to_data,unet_power=3, sample_size=50,
                   epochs=2, image_size=(64,64), random_seed=1,
                   test_split=0.15, batch_size=8, validation_split=0.2)
    upload_model_to_gcp(run_locally=run_locally)
