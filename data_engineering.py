# %%
import os
from tensorflow.keras.datasets import mnist
from PIL import Image
import tensorflow as tf


# %% Prepare MNIST folders and save images of digits 3 and 8
def prepare_mnist_folders(base_dir):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Define training and testing directories
    train_dir = f"{base_dir}/train"
    test_dir = f"{base_dir}/test"

    # Filter and save the dataset for digits 3 and 8
    for x, y, dir in [(X_train, y_train, train_dir), (X_test, y_test, test_dir)]:
        for digit in [3, 8]:
            digit_dir = f"{dir}/{digit}"
            os.makedirs(digit_dir, exist_ok=True)
            for i, img in enumerate(x[y == digit]):
                img_path = f"{digit_dir}/{i}.png"
                Image.fromarray(img).save(img_path)


# Example base directory
base_dir = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/mnist"
prepare_mnist_folders(base_dir)


# %%
class ImageDatasetLoader:
    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size

    def load_dataset(self, dir_path, label_mode="categorical"):
        train_dir = f"{dir_path}/train"
        test_dir = f"{dir_path}/test"

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode=label_mode,
        )

        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode=label_mode,
        )

        return train_dataset, test_dataset


# Initialize the loader with the desired image size and batch size
loader = ImageDatasetLoader(image_size=(224, 224), batch_size=32)

# Load the MNIST dataset
mnist_train, mnist_test = loader.load_dataset(base_dir)

# Assuming you have another dataset in the 'weld' directory
weld_dir = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld"
weld_train, weld_test = loader.load_dataset(weld_dir)

# %%
