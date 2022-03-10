import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def predict_face(fitted_model, image_location, image_size=(64,64)):
    X_test = Image.open(image_location)
    #Resize in case photo not coming from our dataset
    if X_test.size != image_size:
        X_test = X_test.resize((image_size))

    if np.asarray(X_test).ndim == 3:
        return np.squeeze(fitted_model.predict(np.expand_dims(X_test,0)))
    if np.asarray(X_test).ndim == 4:
        return np.squeeze(fitted_model.predict(X_test))
    return "wrong X_test dimension"


def show_predicted_face(model_result):
    """
    plot one image
    """
    return plt.imshow(model_result.astype(np.uint8))
