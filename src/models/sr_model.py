import tensorflow as tf
from tensorflow.keras import layers, models

def srcnn_model(input_shape=(None, None, 1)):
    """
    Defines the architecture of the SRCNN model for super-resolution.
    """
    model = models.Sequential()
    # First convolutional layer with 64 filters, kernel size 9x9
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=input_shape))
    # Second convolutional layer with 32 filters, kernel size 1x1
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    # Third convolutional layer with 1 filter, kernel size 5x5
    model.add(layers.Conv2D(1, (5, 5), activation='linear', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def save_model(model, model_path):
    """
    Saves the trained model to the specified path.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Loads the model from the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model
