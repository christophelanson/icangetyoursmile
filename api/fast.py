from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from PIL import Image
from icangetyoursmile.main import predict_face, show_predicted_face
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import dotenv_values
import matplotlib
import numpy as np
from datetime import datetime
import json
from google.cloud import storage
from google.oauth2 import service_account

settings = dotenv_values() # dictionnary of settings in .env file
BUCKET_NAME='i-can-get-your-smile'
BUCKET_STORAGE_FOLDER='storage'
BUCKET_PREDICTION_FOLDER='predictions'

def get_client(project_id='i-can-get-your-smile', path='keys.json'):
    project_id = 'i-can-get-your-smile'
    with open(path) as source:
        info = json.load(source)
    storage_credentials = service_account.Credentials.from_service_account_info(info)
    client = storage.Client(project=project_id, credentials=storage_credentials)
    return client

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
    client = get_client()
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
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(f'{BUCKET_PREDICTION_FOLDER}/{prediction_name}')
    blob.upload_from_filename(f'./{prediction_name}')
    return blob

def get_image_from_gcp(file_name="seed0001"):
    client = get_client()
    bucket = client.bucket(f"{BUCKET_NAME}")
    blob = bucket.blob(f'to_predict/{file_name}.png')
    blob.download_to_filename('temp.png')
    im = Image.open('temp.png')
    return im

@app.get("/predict")
def predict(image_name, model_name="U-net-christophe",image_size=(64,64)):
    #get model from GCP
    model = download_model_from_gcp(model_name)
    #Get image to predict from file folder
    im = get_image_from_gcp(image_name) #image_name in GCP without ".png"
    #Verify size of image to predict, reshape it if not good size
    if im.size != image_size:
        im = im.resize((image_size))
    #predict
    prediction = np.squeeze(model.predict(np.expand_dims(im,0))).astype(np.uint8)
    #save_prediction(prediction, model_name) #transform&save in the save time
    now = datetime.now()
    current_time = now.strftime("%Hh%M")
    saving_location = f"{model_name}_{image_name}_{current_time}.png"
    matplotlib.image.imsave(saving_location, prediction)
    blob = upload_prediction_to_gcp(saving_location)
    return saving_location #
