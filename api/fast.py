from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import pandas as pd
import joblib
from icangetyoursmile.utils import load_model
from icangetyoursmile.main import predict_face, show_predicted_face

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
def predict(model_name,image_location):
    model = load_model(model_name)
    prediction = predict_face(model, image_location)
    show_predicted_face(prediction)
    #return prediction
    return 'Done'
