from tensorflow.keras.callbacks import Callback
import numpy as np
from icangetyoursmile.trainer import upload_model_to_gcp
from icangetyoursmile.trainer import MODEL_NAME, MODEL_VERSION

class CustomCallback(Callback):

    def __init__(self, images, image_log):
      #  self.model = model
        super().__init__()
        self.images = images # images is a set of 5 images, shape (5,64,64,3) if image size = 64
        self.image_log  = image_log # image_log contains the log of all predicted images, shape (nb_epochs, 5, 64, 64 ,3)

    def on_epoch_end(self, epoch, logs=None):
        """
        appends a list of images with current model.predict
        """
        if epoch <=50 or (epoch<=100 and epoch %2 == 0) or (epoch<200 and epoch %5 == 0) or epoch %10 == 0:
            y_pred = self.model.predict(self.images).astype(np.uint8)
            self.image_log[len(self.image_log)] = y_pred
            print(f" End epoch {epoch}, saved predictions  ")


class SaveModelCallback(Callback):

    def __init__(self, model_name):
      #  self.model = model
        super().__init__()
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):

        if epoch %50 == 0:
            self.model.save(f'./saved_models/{model_name}')
            upload_model_to_gcp(model_name, run_locally=False)
