import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model

def build_sr_model():
    inputs = Input(shape=(None, None, 3))  # Input shape (height, width, channels)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)  # Upsample the feature maps
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer

    model = Model(inputs, x)
    return model

# Build the model
model = build_sr_model()
model.summary()
