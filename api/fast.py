from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

import pytz
import pandas as pd
import joblib

import os

from icangetyoursmile.utils import loading_model
from icangetyoursmile.main import predict_face, show_predicted_face
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import dotenv_values

settings = dotenv_values() # dictionnary of settings in .env file
BUCKET_NAME=settings['BUCKET_NAME']
BUCKET_STORAGE_FOLDER=settings['BUCKET_STORAGE_FOLDER']
BUCKET_PREDICTION_FOLDER=settings['BUCKET_PREDICTION_FOLDER']

import matplotlib
import numpy as np
from datetime import datetime

settings = dotenv_values() # dictionnary of settings in .env file
BUCKET_NAME=settings['BUCKET_NAME']
BUCKET_STORAGE_FOLDER=settings['BUCKET_STORAGE_FOLDER']
BUCKET_PREDICTION_FOLDER=settings['BUCKET_PREDICTION_FOLDER']


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

def download_model_from_gcp(model_name="full-Unet-model"):
    client = storage.Client()
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

# do not work in predict function, def save_prediction(prediction_array, saving_location):
#     matplotlib.image.imsave(saving_location, prediction_array)


def upload_prediction_to_gcp(prediction_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)


    blob = bucket.blob(f'{BUCKET_STORAGE_FOLDER}/{prediction_name}.pickle')
    blob.upload_from_filename(f'./image_logs/{prediction_name}-img_log.pickle')



@app.get("/predict")
def predict(define_prediction_name, image_location, model_name="saved_model"):
    #model = loading_model(model_name)
    #model = load_model("/home/thomast/code/christophelanson/icangetyoursmile/saved_models/2Kimages_150epochs")
    #model = load_model(f"https://storage.cloud.google.com/i-can-get-your-smile/storage/full-Unet-model/{model_name}.pb")
    model = download_model_from_gcp()
    prediction = predict_face(model, image_location)
    saving_location = f"https://storage.cloud.google.com/i-can-get-your-smile/storage/predictions/{define_prediction_name}"
    #need to transform numpy array before saving. use pillow and save .png
    #prediction.save(saving_location)
    #Save & renvoi chemin ou c'est sauvé
    return prediction.shape #{"saving_location": saving_location}

    blob = bucket.blob(f'{BUCKET_PREDICTION_FOLDER}/{prediction_name}')
    blob.upload_from_filename(f'./{prediction_name}')
    return blob


@app.get("/predict")
def predict(image_location, model_name="U-net-christophe"):
    model = download_model_from_gcp(model_name)
    prediction = predict_face(model, image_location).astype(np.uint8)
    #save_prediction(prediction, model_name) #transform&save in the save time
    now = datetime.now()
    current_time = now.strftime("%Hh%M")
    saving_location = f"{model_name}_{current_time}.png"
    matplotlib.image.imsave(saving_location, prediction)
    blob = upload_prediction_to_gcp(saving_location)
    #define full url location, with blob ? to look gcp
    return saving_location #{"saving_location": saving_location}
