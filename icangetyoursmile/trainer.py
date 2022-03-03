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
    blob = bucket.blob(BUCKET_STORAGE_FOLDER)
    # image log
    print('saving image_log')
    blob.upload_from_filename(f'./image_logs/image_log{MODEL_NAME}.pickle')
    print('saving model')
    rel_paths = glob.glob('./saved_models/**', recursive=True)
    for local_path in rel_paths:
        remote_path = f'{BUCKET_STORAGE_FOLDER}{"/".join(local_path.split(os.sep)[1:])}'
        if local_path.isfile(local_path):
            files = os.path.listdir()
            for file in files:
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(file)


if __name__ == '__main__':
    print(FILENAME)
    model_name = MODEL_NAME
    run_locally = True
    path_to_data = None # for notebook use None, else gcp path to data
    run_full_model(model_name, run_locally=run_locally, path_to_data=path_to_data,unet_power=3, sample_size=50,
                   epochs=2, image_size=(64,64), random_seed=1,
                   test_split=0.15, batch_size=8, validation_split=0.2)
    upload_model_to_gcp(run_locally=run_locally)
