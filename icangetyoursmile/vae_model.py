import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from icangetyoursmile.utils import loading_model
import matplotlib.image as mpimg


# Class Sampling

class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder

def encoder(input_shape=(128,128,3),
            latent_dim=64,
            kernel_size=3,
            activation="relu",
            strides=2,
            padding="same"):
    """
    Encoder function to compress the input through 3 layers of convolution.
    Enter an input shape,a kernel size, a latent dimension, a strides and the type of activation and padding
    to be apply for each layer. The filters for the layers are hard coded.
    """
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size, activation=activation, strides=strides, padding=padding)(encoder_inputs)
    x = layers.Conv2D(64, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    x = layers.Conv2D(128, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=activation)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

# Decoder

def decoder(latent_dim=64,
            kernel_size=3,
            activation="relu",
            strides=2,
            padding="same"):
    """
    Decoder function to decompress the input from the encoder function through 4 layers
    of convolution.
    Enter a latent_dimension, a strides and the type of activation and padding
    to be apply for each layer. The filters for the layers are hard-coded.
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation=activation)(latent_inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation=activation, strides=strides, padding=padding)(x)
    decoder_outputs = layers.Conv2DTranspose(3, kernel_size, activation="sigmoid", padding=padding)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

# Define VAE as a model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_absolute_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Creation of the vae_model from the previous functions

def vae_model(encoder=encoder(), decoder=decoder()):
    """
    Create the vae model (variational autoencoder) from the encoder and the decoder.
    The encoder and decoder parameters must be the result from the encoder and decoder function.
    The function returns the vae model compiled with an optimizer Adam.
    """
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    return vae


# Additional functions to help for the usage of the vae model.
# It contains functions for:
#    - Standardizing input for vae model
#    - Plot the 3 differents losses
#    - Recreate model from saved models
#    - Plot original image vs predicted image


def standardize(X):
    """
    Standardize the input(X) for the vae_model by dividing by 255
    """
    X_std = X / 255

    return X_std


def plot_losses_vae(history):
    """
    Plot the loss and reconstruction loss on the same graph, and plot on the side the kl_loss.
    """
    loss = history.history["loss"]
    r_loss = history.history["reconstruction_loss"]
    kl_loss = history.history["kl_loss"]
    epoch_range = range(len(history.history["loss"]))

    #Plot Loss & Reconstruction Loss
    plt.subplot(1,2,1)
    plt.plot(epoch_range, loss, label="Loss")
    plt.plot(epoch_range, r_loss, label="Reconstruction Loss")
    plt.legend(loc="upper right")

    #Plot kl_loss
    plt.subplot(1,2,2)
    plt.plot(epoch_range, kl_loss, label="kl_loss")
    plt.legend(loc="upper right")

    plt.show()


def loaded_model(path_encoder, path_decoder):
    """
    From given paths to encoder and decoder, rebuilt a saved model into a new model
    that can be used for fitting or prediction. Return the new model compiled, the vae_encoder
    and the vae_decoder.
    """
    # Load models
    vae_encoder = loading_model(path_encoder)
    vae_decoder = loading_model(path_decoder)

    # Rebuild model
    new_vae = VAE(vae_encoder, vae_encoder)
    new_vae.compile(optimizer=keras.optimizers.Adam())

    return new_vae, vae_encoder, vae_decoder

def plot_predicted_image(vae_fitted_encoder, vae_fitted_decoder, X, number):
    """
    Plot the original image vs the predicted image from the vae model.
    For the vae encoder and decoder, it's required to entry an already fitted model.

    X is the original input for the vae, not standardize.
    Set an number to display one image from the dataset
    """
    # Standardize X and predict image
    X_std = standardize(X)
    res_encoder = vae_fitted_encoder.predict(X_std)
    res_decoder = vae_fitted_decoder.predict(res_encoder[0])

    # Plot images
    plt.subplot(1,2,1)
    plt.imshow(X[number])
    plt.title("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(res_decoder[number])
    plt.title("Predicted Image")
    plt.show()
