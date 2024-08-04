
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                               epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])

def load_and_preprocess_image(img_path, scale_factor=4):
    """
    Load an image and generate its low-resolution (LR) version on-the-fly.

    :param img_path: Path to the high-resolution (HR) image.
    :param scale_factor: Factor by which to downscale the image to create an LR image.
    :return: Tuple of (LR image, HR image)
    """
    hr_img = img_to_array(load_img(img_path)) / 255.0  # Normalize pixel values
    lr_img = cv2.resize(hr_img, (hr_img.shape[1] // scale_factor, hr_img.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return lr_img, hr_img

def generate_batches(image_dir, batch_size, scale_factor=4):
    """
    Generator that yields batches of LR and HR images.

    :param image_dir: Directory containing the high-resolution images.
    :param batch_size: Number of images per batch.
    :param scale_factor: Factor by which to downscale images to create LR images.
    :yield: Tuple of (batch of LR images, batch of HR images)
    """
    image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]
    while True:
        lr_batch = []
        hr_batch = []
        for _ in range(batch_size):
            img_path = np.random.choice(image_paths)
            lr_img, hr_img = load_and_preprocess_image(img_path, scale_factor)
            lr_batch.append(lr_img)
            hr_batch.append(hr_img)
        yield np.array(lr_batch), np.array(hr_batch)
