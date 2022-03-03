from optparse import Values
from google.cloud import storage
import pandas as pd
import numpy as np

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
MODEL_NAME = 'taxifare'
MODEL_VERSION = 'v1'

def upload_model_to_gcp():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_STORAGE_FOLDER) image_log
    #
    blob.upload_from_filename('model.joblib')











if __name__ == '__main__':
    pass
