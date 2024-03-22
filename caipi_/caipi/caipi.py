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
from skimage.segmentation import mark_boundaries


import random

from keras.models import load_model


class ImageClassifier:
    def __init__(self, model_path, base_folder, target_size=(224, 224)):
        self.model = self.load_pretrained_model(model_path)
        self.base_folder = base_folder
        self.target_size = target_size

    def load_pretrained_model(self, model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model : {e}")
            return None

    def load_and_preprocess_images(self, image_paths):
        images = []
        for img_path in image_paths:
            img = load_img(img_path, target_size=self.target_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            images.append(img)
        return np.array(images)

    def list_image_paths(self):
        image_paths = []
        for root, _, filenames in os.walk(self.base_folder):
            for filename in filenames:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, filename))

        return image_paths

    def select_query(self):
        image_paths = self.list_image_paths()
        images = self.load_and_preprocess_images(image_paths)
        prediction_probs = self.model.predict(images)
        entropies = -np.sum(prediction_probs * np.log(prediction_probs + 1e-5), axis=1)
        uncertain_image_idx = np.argmax(entropies)
        return image_paths[uncertain_image_idx], uncertain_image_idx

    def predict_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = self.model.predict(img_array)
        labels = ["ok", "defect"]
        predicted_label = labels[np.argmax(prediction)]
        return predicted_label, prediction

    def explain_prediction(self, image_path, num_features=5, num_samples=1000):
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_preprocessed = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_preprocessed)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array.astype("double"),
            self.model.predict,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        explanation_image = mark_boundaries(temp / 2 + 0.5, mask)
        return explanation_image


# %%
model_path = "/Users/satyampant/Desktop/Uni/best_model.h5"
base_folder = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/test"

classifier = ImageClassifier(model_path, base_folder)
uncertain_image_path, _ = classifier.select_query()
predicted_label, prediction = classifier.predict_image(uncertain_image_path)
print(
    f"Prediction for the image {uncertain_image_path}: is {prediction},Label is : {predicted_label}"
)

explanation_image = classifier.explain_prediction(uncertain_image_path)

plt.imshow(explanation_image)
plt.title("LIME Explanation")
plt.show()
# %%
