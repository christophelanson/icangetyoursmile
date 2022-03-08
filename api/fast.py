from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import pandas as pd
import joblib
from icangetyoursmile.utils import loading_model
from icangetyoursmile.main import predict_face, show_predicted_face
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import dotenv_values
settings = dotenv_values() # dictionnary of settings in .env file
BUCKET_NAME=settings['BUCKET_NAME']

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

def download_model_from_gcp(model_name="full-Unet-model/"):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(f'storage/{model_name}/')
    blob.download_to_filename('model')
    model = load_model('model')
    return model

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
