
import matplotlib.pyplot as plt
import os
import cv2

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

def preprocess_images(input_dir, output_hr_dir, output_lr_dir, scale_factor=4):
    """
    Preprocess images by creating high-resolution (HR) and low-resolution (LR) pairs.

    :param input_dir: Directory containing the original images.
    :param output_hr_dir: Directory to save the high-resolution images.
    :param output_lr_dir: Directory to save the low-resolution images.
    :param scale_factor: Factor by which to downscale the images to create LR images.
    """
    if not os.path.exists(output_hr_dir):
        os.makedirs(output_hr_dir)
    if not os.path.exists(output_lr_dir):
        os.makedirs(output_lr_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # Save HR image (same as original)
            hr_img = img
            cv2.imwrite(os.path.join(output_hr_dir, img_name), hr_img)

            # Create LR image
            lr_img = cv2.resize(hr_img, (hr_img.shape[1] // scale_factor, hr_img.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC)
            lr_img = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_lr_dir, img_name), lr_img)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_image_data(lr_dir, hr_dir):
    lr_images = []
    hr_images = []

    for img_name in os.listdir(lr_dir):
        lr_img_path = os.path.join(lr_dir, img_name)
        hr_img_path = os.path.join(hr_dir, img_name)

        lr_img = img_to_array(load_img(lr_img_path))
        hr_img = img_to_array(load_img(hr_img_path))

        lr_images.append(lr_img)
        hr_images.append(hr_img)

    lr_images = np.array(lr_images) / 255.0
    hr_images = np.array(hr_images) / 255.0

    return lr_images, hr_images
