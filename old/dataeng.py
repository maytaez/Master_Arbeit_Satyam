# %%
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
dataset_url = "/Users/satyampant/Desktop/proj/dataset"
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
train_good = ["dataset", "train", "good"]
train_bad = ["dataset", "train", "bad"]
test_good = ["dataset", "test", "good"]
test_bad = ["dataset", "test", "bad"]
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
aug_train_good = "/Users/satyampant/Desktop/proj/augemnt/aug_train/good"
# %%
aug_train_bad = "/Users/satyampant/Desktop/proj/augemnt/aug_train/bad"


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
augment(train_good, aug_train_good, 4)  #: 181*4=724
# %%
augment(train_bad, aug_train_bad, 25)  #:28*25

# %%
classes = ["bad", "good"]

for labels in enumerate(classes):
    print(labels)

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# %% Define paths to your training and testing directories
train_dir = "/Users/satyampant/Desktop/proj/augemnt/aug_train"
test_dir = "/Users/satyampant/Desktop/proj/dataset/test"

# Define image data generator with preprocessing
# ResNet expects input images to be 224x224 pixels and preprocess_input will apply the necessary ResNet preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generate batches of tensor image data with real-time data augmentation for the training set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",  # 'binary' if you have binary classes, 'categorical' for multi-class
)
valid_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",  # 'binary' if you have binary classes, 'categorical' for multi-class
)

# %% Generate batches of tensor image data without data augmentation for the testing set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False,  # Do not shuffle the data
    # 'binary' if you have binary classes, 'categorical' for multi-class
)


# %%Load the pre-trained ResNet50 model without the top layer
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)

# Freeze the layers of the base model
base_model.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2(0.01))(
    x
)  # Apply L2 regularization
x = Dropout(0.5)(x)  # Apply Dropout
predictions = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # New output layer

# Define the new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# Callback for early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="min")

# # Callback for learning rate scheduling
# def scheduler(epoch, lr):
#     if epoch < 5:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, min_lr=0.00001
)

# Callback for model checkpointing
checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1, mode="min"
)

# Train the model on your custom dataset
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=100,
    callbacks=[early_stopping, lr_callback, checkpoint],
)

# %% Save the fine-tuned model
# model.save('fine_tuned_resnet_model1.h5')

# %%
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming test_generator and model are already defined as shown in previous steps

# Generate predictions for the test set
predictions = model.predict(test_generator)
# Convert predictions to binary (0 or 1) using 0.5 as a threshold
predicted_classes = np.where(predictions > 0.5, 1, 0).flatten()

# Retrieve true labels from the test_generator
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

# Plotting Accuracy and Loss Graphs
# Assuming you've stored your training history in a variable `history` during model.fit()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.ylim(0.60, 1.00)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# %%
new_model = tf.keras.models.load_model(
    "/Users/satyampant/Desktop/proj/old1/best_model.h5"
)  # same file path


# %%
test_good = list(map(str, (data_dir.glob("test/good/*"))))
test_bad = list(map(str, (data_dir.glob("test/bad/*"))))


# %%
good = True
bad = False
good_matrix = np.array(
    [
        test_good,
        [good] * len(test_good),
        [None] * len(test_good),
        [None] * len(test_good),
    ]
)
good_matrix = np.transpose(good_matrix)

# %%
print(good_matrix[0][0])

# %%
print(good_matrix)
# %%
good = True
bad = False
bad_matrix = np.array(
    [test_bad, [bad] * len(test_bad), [None] * len(test_bad), [None] * len(test_bad)]
)
bad_matrix = np.transpose(bad_matrix)
# %%
print(bad_matrix)
# %%
combined_matrix = np.concatenate((good_matrix, bad_matrix))
print(combined_matrix)
# %%
print(combined_matrix[0][0])
# %%
# i=0
# while i < len(combined_matrix):
#     combined_matrix=new_model.predict(combined_matrix[0])
# print(combined_matrix)
# %%
# %%
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

# import albumentations as al
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Lime package for ML explainability
from lime import lime_image

# for reproducibility (does not guarantee fully reproducible results )
import random

test_dir = "/Users/satyampant/Desktop/proj/dataset/test"
new_model = tf.keras.models.load_model(
    "/Users/satyampant/Desktop/proj/old1/best_model.h5"
)  #
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False,  # Do not shuffle the data
    # 'binary' if you have binary classes, 'categorical' for multi-class
)

# %%

prediction = new_model.predict(test_generator)

# %%
print(prediction)
# %%
predicted_classes = np.where(prediction > 0.5, 1, 0).flatten()

# Retrieve true labels from the test_generator
true_classes = test_generator.classes
# %%
print(true_classes)
# %%
print(test_dir)
# %%
for i in range(len(prediction)):
    if true_classes[i] == 0:
        bad = true_classes[i]
        print(bad)

    else:
        good = true_classes[i]
        print(good)
# %%
matrix = np.array([bad, *len(test_bad), [None] * len(test_bad), [None] * len(test_bad)])
bad_matrix = np.transpose(bad_matrix)

# %%
# Get predictions and class labels
predictions = new_model.predict(test_generator)
class_labels = test_generator.class_indices

# Map labels back to class names (optional)
rev_class_labels = {v: k for k, v in class_labels.items()}

# Iterate through predictions and print results
for i, prediction in enumerate(predictions):
    # Get predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get label and prediction score
    label = rev_class_labels.get(
        predicted_class_index, "Unknown"
    )  # Handle potential missing labels
    score = prediction[predicted_class_index]

    # Print results
    print(f"Image: {test_generator.filenames[i]}")
    print(f"Predicted Label: {label}")
    print(f"Prediction Score: {score:.4f}")  # Format score with 4 decimals
    print("-" * 30)
# %%

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Access original labels from directory structure
original_labels = [path.split("/")[-2] for path in test_generator.filenames]

# Iterate through predictions and print results
for i, (prediction, original_label) in enumerate(zip(predictions, original_labels)):
    # Get predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get prediction score
    score = prediction[predicted_class_index]

    # Print results
    print(f"Image: {test_generator.filenames[i]}")
    print(f"Original Label: {original_label}")
    print(
        f"Predicted Label: {predicted_class_index} ({original_labels[i]})"
    )  # Include original label in predicted label output
    print(f"Prediction Score: {score:.4f}")
    print("-" * 30)
# %%


# Print results
result_matrix = [
    (
        test_generator.filenames[i],
        original_label,
        original_labels[i],
        prediction[np.argmax(prediction)],
    )
    for i, (prediction, original_label) in enumerate(zip(predictions, original_labels))
]


# Define a 4D matrix to store results
# predictions = new_model.predict(test_generator)
# original_labels = [path.split("/")[-2] for path in test_generator.filenames]

# # Create empty matrix to store results
# results_matrix = np.empty((len(predictions), 3))  # 3 columns for label, predicted label, score

# # Fill the matrix with results
# for i, (prediction, original_label) in enumerate(zip(predictions, original_labels)):
#   predicted_class_index = np.argmax(prediction)
#   score = prediction[predicted_class_index]
#   results_matrix[i] = [original_label, predicted_class_index, score]  # Fill each row

# # Print results (optional)
# for row in results_matrix:
#   print(f"Original Label: {row[0]}")
#   print(f"Predicted Label: {int(row[1])} ({row[0]})")  # Cast predicted class index to int for clarity
#   print(f"Prediction Score: {row[2]:.4f}")
#   print("-" * 30)
# %%
print(result_matrix)
# %% selecting random 20 images from test_set
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical
import numpy as np

# %%
# Select 20 random indices from the test set
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(predictions), 20, replace=False)

# Select corresponding predictions and actual labels
selected_predictions = predictions[random_indices]
selected_actual_indices = test_generator.classes[random_indices]
# %%
# Assuming you know the total number of classes `num_classes`
num_classes = len(test_generator.class_indices)

selected_actual_labels_one_hot = to_categorical(
    selected_actual_indices, num_classes=num_classes
)

# %%
# Calculate cosine similarity between actual labels and predictions
cos_similarities = cosine_similarity(
    selected_actual_labels_one_hot, selected_predictions
)

# Print cosine similarities
for i, cos_sim in enumerate(cos_similarities):
    print(f"Image: {test_generator.filenames[random_indices[i]]}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print("-" * 30)

# %%
