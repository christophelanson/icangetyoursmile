from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import pandas as pd
import joblib
from icangetyoursmile.utils import loading_model
from icangetyoursmile.main import predict_face, show_predicted_face
from tensorflow.keras.models import load_model

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

@app.get("/predict")
def predict(image_location):
    #model = loading_model(model_name)
    #model = load_model("/home/thomast/code/christophelanson/icangetyoursmile/saved_models/2Kimages_150epochs")
    model = load_model("saved_models/2Kimages_150epochs")
    prediction = predict_face(model, image_location)
    show_predicted_face(prediction)
    #return prediction
    return prediction.shape
