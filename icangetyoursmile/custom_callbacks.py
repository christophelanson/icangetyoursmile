from tensorflow.keras.callbacks import Callback
import numpy as np

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
        y_pred = self.model.predict(self.images).astype(np.uint8)
        self.image_log[epoch] = y_pred
        print(f" End epoch {epoch}, saved predictions  ")
