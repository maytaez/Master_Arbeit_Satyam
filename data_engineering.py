# %%
import os
from tensorflow.keras.datasets import mnist
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


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

# %% Dataset overview
import numpy as np


class DatasetAnalyzer:
    def __init__(self, datasets_info):
        """
        datasets_info: A dictionary where keys are dataset names and
        values are paths to their respective base directories.
        Example: {'Welding Dataset': '/path/to/quality_dataset', 'MNIST Dataset': '/path/to/mnist_dataset'}
        """
        self.datasets_info = datasets_info

    def count_images(self, base_dir):
        counts = {}

        # Define sub-paths
        sub_paths = ["train", "test"]

        for sub_path in sub_paths:
            path = os.path.join(base_dir, sub_path)
            if not os.path.isdir(path):
                continue

            for category in os.listdir(path):
                category_path = os.path.join(path, category)
                if os.path.isdir(category_path):
                    key = f"{sub_path.capitalize()} {category}"
                    counts[key] = len(
                        [
                            name
                            for name in os.listdir(category_path)
                            if os.path.isfile(os.path.join(category_path, name))
                        ]
                    )

        return counts

    def plot_image_counts(self, dataset_name, counts):
        # Categories and counts
        categories = list(counts.keys())
        counts_values = list(counts.values())

        # Number of bars for each dataset
        n_bars = len(categories)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set position of bar on X axis
        r1 = np.arange(n_bars)

        # Make the plot
        bars = ax.bar(
            r1, counts_values, color=["navy", "darkblue"] * (n_bars // 2), width=0.25
        )

        # Add xticks on the middle of the group bars
        ax.set_xlabel("Category", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_xticks(r1)
        ax.set_xticklabels(categories)
        ax.set_title(dataset_name)

        # Annotate the bars with the count values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def analyze(self):
        # Create a figure for the plots
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(self.datasets_info),
            figsize=(15, 6),
            constrained_layout=True,
        )

        if len(self.datasets_info) == 1:
            axes = [axes]  # Make it iterable

        # Iterate over the given datasets
        for ax, (dataset_name, dataset_path) in zip(axes, self.datasets_info.items()):
            counts = self.count_images(dataset_path)
            categories = list(counts.keys())
            counts_values = list(counts.values())

            # Number of bars for each dataset
            n_bars = len(categories)

            # Set position of bar on X axis
            r1 = np.arange(n_bars)

            # Make the plot
            bars = ax.bar(
                r1,
                counts_values,
                color=["navy", "royalblue"] * (n_bars // 2),
                width=0.50,
            )

            # Add xticks on the middle of the group bars
            ax.set_xlabel("Category", fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
            ax.set_xticks(r1)
            ax.set_xticklabels(categories, rotation=45)
            ax.set_title(dataset_name)

            # Annotate the bars with the count values
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    "{}".format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        # Adjust layout to make room for category labels
        plt.tight_layout()
        plt.show()


# Example usage
datasets_info = {
    "Welding Dataset": "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld",
    "MNIST Dataset": "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/mnist",
}

analyzer = DatasetAnalyzer(datasets_info)
analyzer.analyze()


# %% Image Augmentation on Welding Dataset
import os
import pathlib
import PIL

import cv2
import skimage
from IPython.display import Image, display
from matplotlib.image import imread
import matplotlib.cm as cm

# Tensorflow basics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import albumentations as al
import numpy as np


# Lime package for ML explainability
from lime import lime_image

# for reproducibility (does not guarantee fully reproducible results )
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# %%
# %%
dataset_url = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld"
data_dir = pathlib.Path(dataset_url)
data_dir
# %% Total images inside dataset(train + test)
image_count = len(list(data_dir.glob("**/*.jpg")))
print(image_count)
# %%
bad = list(data_dir.glob("train/bad/*"))
PIL.Image.open(bad[0])
# %%
good = list(data_dir.glob("train/good/*"))
PIL.Image.open(good[0])
# %%shape of images
sample = imread(bad[0])
sample.shape

# %%
train_good = [
    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld",
    "train",
    "good",
]
train_bad = ["/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld", "train", "bad"]
test_good = ["/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld", "test", "good"]
test_bad = ["/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld", "test", "bad"]
# %%
owd = os.getcwd()
train_good = os.path.join(owd, *train_good)
train_bad = os.path.join(owd, *train_bad)
test_good = os.path.join(owd, *test_good)
test_bad = os.path.join(owd, *test_bad)

print(f"Train Good Images Directory: {train_good}")
print(f"Train Bad Images Directory: {train_bad}")
print(f"Test Good Images Directory: {test_good}")
print(f"Test Bad Images Directory: {test_bad}")
# %%
for infile in os.listdir(train_good):
    print(infile)
# %%Image Pre-Processing and Augmentation

## Resizing Images
HEIGHT = 224
WIDTH = 224
transform = al.Compose(
    [
        al.Resize(width=WIDTH, height=HEIGHT, p=1.0),
        al.Rotate(limit=90, p=1.0),
        al.HorizontalFlip(p=0.3),
        al.VerticalFlip(p=0.2),
        al.ColorJitter(contrast=2, p=0.2),
        al.ColorJitter(brightness=2, p=0.2),
        al.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, p=1.0
        ),
    ]
)
# %% storing images after augmenting
aug_train_good = (
    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/aug_train/good"
)
# %%
aug_train_bad = (
    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/aug_train/bad"
)


# %%Augmenting bad for 30 times
def augment(IMG_DIR, AUG_PATH_IMAGE, num):
    print("*******************Augmentation Started*****************************")
    for i, infile in enumerate(os.listdir(IMG_DIR)):
        image = cv2.imread(os.path.join(IMG_DIR, infile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for j in range(num):
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            transformed_image = transformed_image * 255
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            fname = AUG_PATH_IMAGE + "/" + infile[:-4] + "_" + str(j) + ".jpg"
            cv2.imwrite(fname, transformed_image)
    print()
    print("*******************Augmentation Finished*****************************")


# %%
# augment(train_good, aug_train_good, 4)  #: 181*4=724
augment(train_good, aug_train_good, 20)  #: 181*20=3620
# %%
# augment(train_bad, aug_train_bad, 25)  #:28*25
augment(train_bad, aug_train_bad, 125)  #:28*125=3500

# %%
classes = ["bad", "good"]

for labels in enumerate(classes):
    print(labels)

# %% Model engineering
# finetuning model according to both the datasets and performing grid search on parameters like learning rate and batch size and save the best model for each dataset, Also track the model performance on the hyperparameter and save in csv or json format the results.

import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


class ModelTrainer:
    def __init__(self, datasets_info):
        self.datasets_info = datasets_info

    def build_model(self):
        base_model = ResNet50(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        for layer in base_model.layers:
            layer.trainable = False

        global_average_layer = GlobalAveragePooling2D()
        prediction_layer = Dense(1, activation="sigmoid")
        model = tf.keras.Sequential(
            [
                base_model,
                global_average_layer,
                prediction_layer,
            ]
        )
        return model

    def prepare_data(self, dataset_path, validation_split=0.2, batch_size=32):
        train_data_gen = ImageDataGenerator(
            rescale=1.0 / 255, validation_split=validation_split
        )
        train_generator = train_data_gen.flow_from_directory(
            os.path.join(dataset_path, "train"),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
            subset="training",
        )
        validation_generator = train_data_gen.flow_from_directory(
            os.path.join(dataset_path, "train"),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
            subset="validation",
        )
        test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_data_gen.flow_from_directory(
            os.path.join(dataset_path, "test"),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )
        return train_generator, validation_generator, test_generator

    def fine_tune_model(
        self,
        model,
        train_generator,
        validation_generator,
        learning_rate=0.001,
        epochs=100,
    ):
        for layer in model.layers[:100]:  # Fine-tune from a specific layer onwards
            layer.trainable = True

        model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
            ),
            ModelCheckpoint(
                "best_model.h5",
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode="max",
            ),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ]

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
        )

        return history

    def grid_search(self, param_grid):
        results = []
        for params in ParameterGrid(param_grid):
            for dataset_name, dataset_path in self.datasets_info.items():
                train_generator, validation_generator, test_generator = (
                    self.prepare_data(dataset_path, batch_size=params["batch_size"])
                )
                model = self.build_model()
                history = self.fine_tune_model(
                    model,
                    train_generator,
                    validation_generator,
                    learning_rate=params["learning_rate"],
                    epochs=params["epochs"],
                )
                val_accuracy = history.history["val_accuracy"][-1]
                results.append(
                    {
                        "dataset": dataset_name,
                        "params": params,
                        "val_accuracy": val_accuracy,
                    }
                )
        return results

    def save_best_models(self, results, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for result in results:
            dataset_name = result["dataset"]
            params = result["params"]
            model = self.build_model()
            train_generator, validation_generator, test_generator = self.prepare_data(
                self.datasets_info[dataset_name], batch_size=params["batch_size"]
            )
            self.fine_tune_model(
                model,
                train_generator,
                validation_generator,
                learning_rate=params["learning_rate"],
                epochs=params["epochs"],
            )
            model.save(os.path.join(save_dir, f"{dataset_name}_best_model.h5"))

    def save_results(self, results, save_path):
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path, index=False)


# Example usage
datasets_info = {
    "Dataset 1": "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/mnist",
    "Dataset 2": "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld",
}

trainer = ModelTrainer(datasets_info)

# Define hyperparameter grid
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128],
    "epochs": [5, 10],
}

# Perform grid search
results = trainer.grid_search(param_grid)

# Save results
trainer.save_results(results, "results.csv")

# Save best models
trainer.save_best_models(results, "best_models")
