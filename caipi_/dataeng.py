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

import shap
from operator import itemgetter


import random

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
import json


# %%
dataset_url = "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_data/"
train_path = dataset_url + "train/"
test_path = dataset_url + "test/"
data_dir = pathlib.Path(dataset_url)
data_dir
# # %%
# image_count = len(list(data_dir.glob('*/*.jpeg')))
# print(image_count)
# # %% defective image
# def_front = list(data_dir.glob('def_front/*'))
# PIL.Image.open(def_front[0])
# # %% non-defective image
# ok_front = list(data_dir.glob('ok_front/*'))
# PIL.Image.open(ok_front[0])

# # %% shape of images
# sample1=imread(ok_front[0])
# sample1.shape
# %%
plt.figure(figsize=(10, 8))
ok = plt.imread(train_path + "ok_front/cast_ok_0_1.jpeg")
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("ok", weight="bold", size=20)
plt.imshow(ok, cmap="gray")

ng = plt.imread(train_path + "def_front/cast_def_0_1001.jpeg")
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("def", weight="bold", size=20)
plt.imshow(ng, cmap="gray")

plt.show()
# %%
img = cv2.imread(train_path + "ok_front/cast_ok_0_1.jpeg")
img_4d = img[np.newaxis]
plt.figure(figsize=(25, 10))
generators = {
    "rotation": ImageDataGenerator(rotation_range=180),
    "zoom": ImageDataGenerator(zoom_range=0.7),
    "brightness": ImageDataGenerator(brightness_range=[0.2, 1.0]),
    "height_shift": ImageDataGenerator(height_shift_range=0.7),
    "width_shift": ImageDataGenerator(width_shift_range=0.7),
}

plt.subplot(1, 6, 1)
plt.title("Original", weight="bold", size=15)
plt.imshow(img)
plt.axis("off")
cnt = 2
for param, generator in generators.items():
    image_gen = generator
    gen = image_gen.flow(img_4d, batch_size=1)
    batches = next(gen)
    g_img = batches[0].astype(np.uint8)
    plt.subplot(1, 6, cnt)
    plt.title(param, weight="bold", size=15)
    plt.imshow(g_img)
    plt.axis("off")
    cnt += 1
plt.show()
# %%Data Preparation for model training
batch_size = 32
epochs = 100
img_height = 224
img_width = 224
img_size = (img_height, img_width)
# %%
image_gen = ImageDataGenerator(
    rescale=1 / 255, zoom_range=0.1, brightness_range=[0.9, 1.0]
)

# %% splitting data 80% train and 20% valid
train_set = keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    class_names=["ok_front", "def_front"],
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    shuffle=True,
    batch_size=batch_size,
)

# %%
val_set = keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    class_names=["ok_front", "def_front"],
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
# %%
test_set = keras.utils.image_dataset_from_directory(
    test_path,
    class_names=["ok_front", "def_front"],
    seed=seed,
    image_size=(img_height, img_width),
    shuffle=False,
    batch_size=batch_size,
)
# %%checking the class names
class_names = train_set.class_names
print(class_names)

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# %%(32-batch size, 224,224-height,width, 3-RGB)
for images, labels in train_set:
    print(images.shape)
    print(labels.shape)
    break
# %%
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_set.cache().shuffle(1300).prefetch(buffer_size=AUTOTUNE)
val_ds = val_set.cache().prefetch(buffer_size=AUTOTUNE)
# %%
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal_and_vertical", input_shape=(img_height, img_width, 3), seed=seed
        ),
        layers.RandomZoom(0.1, seed=seed),
        layers.RandomContrast(0.3, seed=seed),
    ]
)
# %%
custom_model = Sequential(
    [
        layers.Rescaling(1.0 / 255),
        data_augmentation,
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
# %%
custom_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
# IMAGE_SIZE=224
# NUM_CHANNELS=3
# custom_model.build(input_shape=(None, IMAGE_SIZE, NUM_CHANNELS))

# plot_model(custom_model, show_shapes=True, expand_nested=True, dpi=60)


# %%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") == 1.0 and logs.get("val_accuracy") == 1.0:
            print("\nReached 100% accuracy so cancelling training!")
            self.model.stop_training = True


terminate_callback = myCallback()
# %%
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    verbose=1,
    min_delta=0.01,
    patience=5,
    min_lr=0.000001,
)
# %%
history1 = custom_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[reduce_lr, terminate_callback],
)

# %%
custom_model.summary()
# %%
acc = history1.history["accuracy"]
val_acc = history1.history["val_accuracy"]

loss = history1.history["loss"]
val_loss = history1.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# %%Using Transfer Learning

base_model = keras.applications.ResNet50(
    weights="imagenet", input_shape=(img_height, img_width, 3), include_top=False
)
# %%
base_model.trainable = True

# %% Freeze all layers except the last few
for layer in base_model.layers[:-10]:
    layer.trainable = False
# %%
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9
)
# Create new model on top
inputs = keras.Input(shape=(img_height, img_width, 3))

x = data_augmentation(inputs)  # Apply random data augmentation

x = keras.layers.Rescaling(scale=1 / 224.0)(x)

x = base_model(x, training=False)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
pretrained_model = keras.Model(inputs, outputs)
# %%
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
pretrained_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
# %%
model_save_path = "casting_product_detection.hdf5"
early_stop = EarlyStopping(monitor="val_loss", patience=2)
checkpoint = ModelCheckpoint(
    filepath=model_save_path, verbose=1, save_best_only=True, monitor="val_loss"
)
# %%
history2 = pretrained_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop, terminate_callback, checkpoint],
)
# %%
model_history = {
    i: list(map(lambda x: float(x), j)) for i, j in history2.history.items()
}
with open("model_history.json", "w") as f:
    json.dump(model_history, f, indent=4)

# %%
losses = pd.DataFrame(model_history)
losses.index = map(lambda x: x + 1, losses.index)
losses.head(10)
# %%
acc = history2.history["accuracy"]
val_acc = history2.history["val_accuracy"]

loss = history2.history["loss"]
val_loss = history2.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# %%Confusion Matrix
import seaborn as sns

true_labels = []
for images, labels in test_set.unbatch().take(
    -1
):  # take(-1) will take all the data in the dataset
    true_labels.extend(labels.numpy())

# Now, make predictions using the test dataset
pred_probability = pretrained_model.predict(test_set)
predictions = pred_probability > 0.5

# Flatten the predictions to match the true labels' shape
predictions = np.argmax(predictions, axis=-1)

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 6))
plt.title("Confusion Matrix", size=20, weight="bold")
sns.heatmap(
    cm,
    annot=True,
    annot_kws={"size": 14, "weight": "bold"},
    fmt="d",
    cmap="Blues",
    xticklabels=test_set.class_indices.keys(),
    yticklabels=test_set.class_indices.keys(),
)

plt.tick_params(axis="both", labelsize=14)
plt.ylabel("Actual", size=14, weight="bold")
plt.xlabel("Predicted", size=14, weight="bold")
plt.show()


# %%
# Model Explainabilitz
# img_path = '/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_data/test/def_front/cast_def_0_143.jpeg'
img_path = "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_data/test/def_front/cast_def_0_520.jpeg"
display(Image(img_path))


# %%
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


# %%
images = get_img_array(img_path, img_size)
preds = pretrained_model.predict(images)
prediction = np.argmax(preds)
pct = np.max(preds)
print(pct)
# %%
# if prediction == 1:
#     print("ok")
# else:
#     print("defect")

# print(pct)
predicted_class_prob = preds[0, 0]
print(predicted_class_prob)

# Define a threshold (e.g., 0.5) to determine the class label
threshold = 0.5

# Determine the predicted class based on the threshold
predicted_class_label = "ok" if predicted_class_prob >= threshold else "defect"

# Print the predicted class label
print(f"Predicted Class: {predicted_class_label}")
# %%

explainer = lime_image.LimeImageExplainer()

# %% during this process a local model is being created to explain why this prediction is refered to be the result class with this much accuracy.

explanation = explainer.explain_instance(
    images[0].astype("double"),
    pretrained_model.predict,
    top_labels=2,
    hide_color=255,
    num_samples=300,
)
# %%
explanation
# %% to show outline boundaries for specific areas in the given image
from skimage.segmentation import mark_boundaries

# %% Creating superpixel for the prediction(showing which are was responsible for predicting the image as the correct label)
temp_1, mask_1 = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
)
plt.imshow(mark_boundaries(temp_1 / 2 + 0.5, mask_1))
# %%
temp_1, mask_1 = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
)
plt.imshow(mark_boundaries(temp_1 / 2 + 0.5, mask_1))

# %%Visualizing pros in green and cons in red

temp_1, mask_1 = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
)
plt.imshow(mark_boundaries(temp_1 / 2 + 0.5, mask_1))
# %%showing positives and negatives having some sort of significance.
# removing positives and negatives having 0.1 %.
temp_1, mask_1 = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False,min_weight=0.1
)
plt.imshow(mark_boundaries(temp_1 / 2 + 0.5, mask_1))

# %% Explaination Heatmap plot with weights
#The colorbar shows the values of the weights

ind=explanation.top_labels[0]

#Mapping each explanation weight to the corresponding superpixel
dict_heatmap=dict(explanation.local_exp[ind])
heatmap=np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot 
plt.imshow(heatmap,cmap='RdBu',vmin=-heatmap.max(),vmax=heatmap.max())
plt.colorbar()
# %%
from keras.applications.imagenet_utils import decode_predictions,preprocess_input

#%%
img_array=keras.applications.resnet50.preprocess_input(images)
predictions=pretrained_model.predict(img_array)

# %%
# Decode the predictions into human-readable labels
decoded_predictions = decode_predictions(predictions, top=2)  
print(decode_predictions)
# Print the top predictions
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

