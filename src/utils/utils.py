import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def load_image(image_path):
    """Loads and decodes an image from the file system."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def preprocess_image(image, target_size):
    """Resizes the image to the given target size."""
    image = tf.image.resize(image, target_size)
    return image

def load_and_preprocess_image(image_path, low_res_size, high_res_size):
    """Loads an image, creates low-res and high-res versions, and returns them."""
    img = load_image(image_path)
    high_res_img = preprocess_image(img, high_res_size)  # Оригинальное изображение
    low_res_img = preprocess_image(img, low_res_size)  # Создаем низкорезолюционное изображение
    low_res_img = preprocess_image(low_res_img, high_res_size)  # Увеличиваем до оригинального размера
    return low_res_img, high_res_img

def load_dataset(root_dir, low_res_size, high_res_size):
    """Creates a TensorFlow dataset that yields pairs of low-res and high-res images."""
    image_files = []
    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                image_files.append(os.path.join(person_dir, img_name))

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(lambda x: load_and_preprocess_image(x, low_res_size, high_res_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def load_and_show_images(extracted_dir, num_examples=5):
    plt.figure(figsize=(15, 8))
    count = 0
    for person_name in os.listdir(extracted_dir):
        person_dir = os.path.join(extracted_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, image_name)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.subplot(1, num_examples, count + 1)
                plt.imshow(img_rgb)
                plt.title(person_name)
                plt.axis('off')

                count += 1
                if count == num_examples:
                    break
            if count == num_examples:
                break
    plt.show()



def plot_training_history(history):
    """
    Plots the training and validation loss and metrics over the epochs.

    Args:
    history (History): History object returned by the fit method of a Keras model.
    """

    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot PSNR
    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr_metric'], label='Training PSNR')
    if 'val_psnr_metric' in history.history:
        plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
    plt.title('PSNR over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()

    plt.show()
