# %%
import os
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from lime import lime_image
import tkinter as tk
from skimage.segmentation import mark_boundaries
from tkinter import filedialog


class ImageClassifierApp:
    def __init__(self, model_path):
        self.model = self.load_pretrained_model(model_path)
        self.window = tk.Tk()
        self.window.title("Image Classifier with LIME Explanation")
        self.uncertain_image_path = "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_data/test/ok_front/cast_ok_0_16.jpeg"
        self.setup_ui()

    def load_pretrained_model(self, model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def setup_ui(self):
        self.load_button = tk.Button(
            self.window, text="Load Image", command=self.load_image
        )
        self.load_button.pack()

        self.predict_button = tk.Button(
            self.window, text="Predict Image", command=self.predict_image_and_explain
        )
        self.predict_button.pack()

        self.explain_button = tk.Button(
            self.window, text="Explain Prediction", command=self.explain_prediction
        )
        self.explain_button.pack()

        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        self.prediction_label = tk.Label(self.window)
        self.prediction_label.pack()

        self.explanation_label = tk.Label(self.window)
        self.explanation_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.uncertain_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((224, 224), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def predict_image_and_explain(self):
        if self.uncertain_image_path and os.path.exists(self.uncertain_image_path):
            predicted_label, prediction = self.predict_image(self.uncertain_image_path)
            self.prediction_label.config(
                text=f"Prediction: {predicted_label}, Confidence: {np.max(prediction)}"
            )
        else:
            self.prediction_label.config(
                text="No image loaded or image path is incorrect"
            )

    def predict_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)
        labels = ["ok", "defect"]
        predicted_label = labels[np.argmax(prediction)]
        return predicted_label, prediction

    def explain_prediction(self):
        if self.uncertain_image_path and os.path.exists(self.uncertain_image_path):
            try:
                explanation_image = self.generate_lime_explanation(
                    self.uncertain_image_path
                )
                explanation_image = Image.fromarray(
                    (explanation_image * 255).astype("uint8")
                )
                explanation_photo = ImageTk.PhotoImage(explanation_image)
                self.explanation_label.config(image=explanation_photo)
                self.explanation_label.image = explanation_photo  # Keep a reference
                self.window.update_idletasks()  # Update UI
            except Exception as e:
                self.explanation_label.config(text=f"Error in explanation: {e}")
        else:
            self.explanation_label.config(
                text="No image loaded or image path is incorrect"
            )

    def generate_lime_explanation(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_preprocessed = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_preprocessed)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array.astype("double"),
            self.model.predict,
            top_labels=2,
            hide_color=255,
            num_samples=1000,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=2,
            hide_rest=False,
        )
        explanation_image = mark_boundaries(temp / 2 + 0.5, mask)

        return explanation_image

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = ImageClassifierApp(
        "/Users/satyampant/Desktop/Master_Arbeit_Satyam/casting_product_detection.hdf5"
    )
    app.run()
# %%
