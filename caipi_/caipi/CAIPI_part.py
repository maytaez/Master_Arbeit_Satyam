# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pathlib
import PIL
import cv2
import skimage
from IPython.display import Image, display
from matplotlib.image import imread
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.optimizers import Adam
from lime import lime_image
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import shap
from operator import itemgetter


import random

# %% CAIPI part

from keras.models import load_model


def load_pretrained_model(model_path):
    model = load_model(model_path)
    return model


# %%Using loaded model to make a prediction on input data x
def predict(model, x):
    y_hat = model.predict(x)
    return y_hat


# %% Select Query
# Query selection part aims to find the most informative examples from the unlabelled dataset for the model to learn from.
# These are the examples where the model is most uncertain about the predictions
# This technique is called uncertainity sampling

# load and preprocess the image


def load_and_preprocess(image_path, target_size):
    images = []
    for img_path in image_path:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        img = preprocess_input(img)
        images.append(img)
    # Converting list to a numpy array and return
    return np.array(images)


# %%Getting list of image paths from the subfolders
def list_image_paths(base_folder):
    image_paths = []
    for root, _, filenames in os.walk(base_folder):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, filename))
    return image_paths


# %% the main select querry function


def select_query(model, base_folder, target_size=(224, 224)):
    # list of all image paths in base folder
    image_paths = list_image_paths(base_folder)

    # Load and preprocess images
    images = load_and_preprocess(image_paths, target_size)

    # Get model's prediction probabilities for each image
    prediction_probs = model.predict(images)

    # Calculating uncertainity for each image
    entropies = -np.sum(prediction_probs * np.log(prediction_probs + 1e-5), axis=1)

    # Selecting index with highest entropy
    uncertain_image_idx = np.argmax(entropies)

    # Return most uncertain image path and its index

    return image_paths[uncertain_image_idx], uncertain_image_idx


# %%Predict unlabelled images from image_paths using the pretrtained model
def predict_image(model, image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)

    labels = ["ok", "defect"]
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label, prediction


# usage
model_path = (
    "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_product_detection.hdf5"
)
model = load_pretrained_model(model_path)

base_folder = "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_data/test"
target_size = (224, 224)
uncertain_image_path, _ = select_query(model, base_folder, target_size)

# Predict the selected image
predicted_label, prediction = predict_image(model, uncertain_image_path, target_size)

# Output the prediction
print(
    f"Prediction for the image {uncertain_image_path}: is {prediction}, Label is: {predicted_label} "
)

# %% Using LIME to explain the image
from skimage.segmentation import mark_boundaries


def explain_prediction(
    model, uncertain_image_path, target_size, num_features=5, num_samples=1000
):
    model = load_pretrained_model(model_path)
    img = load_img(uncertain_image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_preprocessed)

    # Explaining pred using lime
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype("double"),
        model.predict,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
    )

    # Get image and mask for the explanation

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    explanation_image = mark_boundaries(temp / 2 + 0.5, mask)

    return explanation_image


# %%
model_path = (
    "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_product_detection.hdf5"
)
model = load_pretrained_model(model_path)
uncertain_image_path, _ = select_query(model, base_folder, target_size)
target_size = (224, 224)

# Generate explanation for prediction
explanation_image = explain_prediction(model, uncertain_image_path, target_size)


# %%
