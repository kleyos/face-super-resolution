import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model


def build_sr_model(input_shape=(128, 128, 3)):
    """
    Build a simple super-resolution model.

    :param input_shape: Shape of the input low-resolution image.
    :return: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = UpSampling2D(size=(2, 2))(x)  # Upscale to 256x256
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)  # Upscale to 512x512
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer, should match HR dimensions

    model = Model(inputs, x)
    return model
