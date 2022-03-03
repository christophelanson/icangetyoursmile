from models import *


def predict_face(fitted_model, X_test):
    """
    return model.predict
    if X_test is a single image, dim = (64,64,3) if img size =64
    if X_test is a set of images, dim = (None,64,64,3) if img size = 64
    """
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
