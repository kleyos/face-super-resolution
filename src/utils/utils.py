
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator


def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    # Plot training & validation loss values
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, h['loss'], 'b', label='Training loss')
    plt.plot(epochs, h['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def generate_batches(hr_dir, batch_size, lr_target_size=(128, 128), hr_target_size=(512, 512), scale=4):
    """
    Generator that yields low-resolution and high-resolution image pairs on-the-fly.

    :param hr_dir: Directory containing high-resolution images.
    :param batch_size: Number of images per batch.
    :param lr_target_size: Target size for low-resolution images.
    :param hr_target_size: Target size for high-resolution images.
    :param scale: Downscale factor for generating low-resolution images.
    :return: Yields batches of (low-resolution images, high-resolution images).
    """
    datagen = ImageDataGenerator(rescale=1./255)

    hr_generator = datagen.flow_from_directory(
        hr_dir,
        target_size=hr_target_size,  # Load HR images at the original size
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        classes=['.']
    )

    while True:
        hr_batch = hr_generator.next()

        # On-the-fly downscale to create low-resolution images
        lr_batch = np.array([cv2.resize(img, lr_target_size, interpolation=cv2.INTER_CUBIC) for img in hr_batch])

        yield lr_batch, hr_batch