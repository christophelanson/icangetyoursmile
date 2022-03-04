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
import matplotlib
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

def save_prediction(prediction_array, model_name):
    matplotlib.image.imsave(f'{model_name}.png', prediction_array)


def upload_prediction_to_gcp(prediction_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(f'{BUCKET_PREDICTION_FOLDER}/{prediction_name}')
    blob.upload_from_filename(f'./{prediction_name}')



@app.get("/predict")
def predict(define_prediction_name, image_location, model_name="full-Unet-model"):
    model = download_model_from_gcp(model_name)
    prediction = predict_face(model, image_location)
    #save_prediction(prediction, model_name) #transform&save in the save time
    saving_location = f"{model_name}.png"
    #need to transform numpy array before saving. use pillow and save .png
    upload_prediction_to_gcp(saving_location)
    #define full url location, with blob ? to look gcp
    return prediction.shape #{"saving_location": saving_location}
