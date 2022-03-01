from models import *


def predict_face(fitted_model, X_test):
    if np.asarray(X_test).ndim == 3:
        return np.squeeze(fitted_model.predict(np.expand_dims(X_test,0)))
    if np.asarray(X_test).ndim == 4:
        return np.squeeze(fitted_model.predict(X_test))
    return "wrong X_test dimension"

def show_predicted_face(model_result):
    return plt.imshow(model_result.astype(np.uint8))