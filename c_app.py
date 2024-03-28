# %%
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageGrab, ImageOps
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import rescale, rotate, AffineTransform, warp
import os
from PIL import Image, ImageDraw, ImageChops


class ImageClassifierApp:
    def __init__(self, model_path, base_folder):
        self.model = self.load_pretrained_model(model_path)
        self.base_folder = base_folder
        self.window = tk.Tk()
        self.window.title("Image Classifier with LIME Explanation")
        self.target_size = (224, 224)
        self.draw_color = "green"
        self.eraser_on = False
        self.mask = None
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

        self.confirm_prediction_button = tk.Button(
            self.window, text="Confirm Prediction", command=self.confirm_prediction
        )
        self.confirm_prediction_button.pack()

        self.confirm_explanation_button = tk.Button(
            self.window, text="Confirm Explanation", command=self.confirm_explanation
        )
        self.confirm_explanation_button.pack()

        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        self.prediction_label = tk.Label(self.window)
        self.prediction_label.pack()

        # self.augment_button = tk.Button(
        #     self.window, text="Augment Image", command=self.augment_image
        # )
        # self.augment_button.pack()
        self.save_button = tk.Button(
            self.window,
            text="Save Annotated Image",
            command=self.save_drawn_annotations,
        )
        self.canvas = tk.Canvas(self.window, width=224, height=224)
        self.canvas.pack()

        # self.toggle_eraser_button = tk.Button(
        #     self.window, text="Toggle Eraser", command=self.toggle_eraser
        # )
        # self.toggle_eraser_button.pack()

        self.save_button = tk.Button(
            self.window, text="Save Annotated Image", command=self.save_annotated_image
        )
        self.save_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.uncertain_image_path = file_path
            img = Image.open(file_path)
            img = img.resize(self.target_size, Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def predict_image_and_explain(self):
        if hasattr(self, "uncertain_image_path") and os.path.exists(
            self.uncertain_image_path
        ):
            predicted_label, prediction = self.predict_image(self.uncertain_image_path)
            self.prediction_label.config(
                text=f"Prediction: {predicted_label}, Confidence: {np.max(prediction)}"
            )

            explanation_image = self.explain_prediction(self.uncertain_image_path)
            self.display_explanation_on_canvas(explanation_image)
        else:
            self.prediction_label.config(
                text="No image loaded or image path is incorrect"
            )

    def predict_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = self.model.predict(img_array)
        labels = ["bad", "good"]
        predicted_label = labels[np.argmax(prediction)]
        return predicted_label, prediction

    def explain_prediction(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array.astype("double"),
            self.model.predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        explanation_image = mark_boundaries(temp / 2 + 0.5, mask)
        explanation_image = Image.fromarray((explanation_image * 255).astype("uint8"))
        return explanation_image

    def display_explanation_on_canvas(self, explanation_image):
        self.explained_photo = ImageTk.PhotoImage(explanation_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.explained_photo)
        self.canvas.image = self.explained_photo  # Keep reference

    def confirm_prediction(self):
        response = messagebox.askyesno(
            "Confirm Prediction", "Is the prediction correct?"
        )
        if response:
            messagebox.showinfo(
                "Prediction Confirmation", "Prediction confirmed as correct."
            )
        else:
            self.change_prediction_label()

    def change_prediction_label(self):
        new_label = simpledialog.askstring("Input", "Enter the correct label:")
        if new_label:
            self.corrected_label = new_label
            self.prediction_label.config(text=f"Corrected Label: {new_label}")

    def confirm_explanation(self):
        response = messagebox.askyesno(
            "Confirm Explanation", "Is the explanation correct?"
        )
        if not response:
            # If the explanation is wrong, enable drawing on the orignal image
            self.display_orignal_image_on_canvas()
            self.canvas.bind("<B1-Motion>", self.draw_on_image)
            # self.toggle_eraser_button.pack()  # Show the eraser toggle button

    def display_orignal_image_on_canvas(self):
        if hasattr(self, "uncertain_image_path") and os.path.exists(
            self.uncertain_image_path
        ):
            img = Image.open(self.uncertain_image_path)
            # img=img.resize(self.uncertain_image_path)
            img = img.resize(self.target_size, Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.canvas.image = self.photo

    def scale_image(self, image, scale_factor=1.2):
        return rescale(image, scale_factor, anti_aliasing=True, multichannel=True)

    def rotate_image(self, image, angle=45):
        return rotate(image, angle)

    def translate_image(self, image, translation=(25, 25)):
        transform = AffineTransform(translation=translation)
        return warp(image, transform, mode="wrap")

    def draw_on_image(self, event):
        x, y = event.x, event.y
        size = 10 if self.eraser_on else 5
        if self.mask is None:
            self.mask = Image.new(
                "L", (224, 224), 0
            )  # Create a new mask if it doesn't exist
        draw = ImageDraw.Draw(self.mask)
        if self.eraser_on:
            # Update both the canvas and the mask for erasing
            self.canvas.create_rectangle(
                x - size, y - size, x + size, y + size, outline="", fill="white"
            )
            draw.rectangle([x - size, y - size, x + size, y + size], fill=0)
        else:
            # Update both the canvas and the mask for drawing
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size, fill="black", outline="black"
            )
            draw.ellipse([x - size, y - size, x + size, y + size], fill=255)

    def save_annotated_image(self):
        if self.mask is not None:
            original_img = Image.open(self.uncertain_image_path).convert("RGB")
            # Ensure the mask is the same size as the original image
            scaled_mask = self.mask.resize(original_img.size, Image.ANTIALIAS)
            # Apply the mask to the original image
            result_img = ImageChops.multiply(original_img, scaled_mask.convert("RGB"))
            result_img_path = os.path.join(self.base_folder, "annotated_img.png")
            result_img.save(result_img_path)
            messagebox.showinfo(
                "Success", f"Annotated image saved successfully at {result_img_path}"
            )
        else:
            messagebox.showerror("Error", "No annotations to save.")

    def save_drawn_annotations(self):
        # Get the size of the original image
        original_img = Image.open(self.uncertain_image_path).convert("RGB")

        # Create an empty mask with the same dimensions as the original image
        mask = Image.new("L", original_img.size, 0)
        draw = ImageDraw.Draw(mask)

        # Assume the canvas drawing coordinates are stored in a list of tuples
        drawing_coords = [(x1, y1, x2, y2)]  # Replace with the actual coordinates

        # Scale the drawing coordinates to the size of the original image
        scale_width = original_img.width / self.target_size[0]
        scale_height = original_img.height / self.target_size[1]

        scaled_coords = [
            (x1 * scale_width, y1 * scale_height, x2 * scale_width, y2 * scale_height)
            for (x1, y1, x2, y2) in drawing_coords
        ]

        # Draw the scaled coordinates onto the mask
        for x1, y1, x2, y2 in scaled_coords:
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # Threshold the mask to make sure it's binary
        thresholded_mask = mask.point(lambda x: 255 if x > 0 else 0, "1")

        # Crop the original image using the mask
        # The mask defines the area to keep
        bbox = thresholded_mask.getbbox()
        cropped_image = original_img.crop(bbox)
        cropped_image.save("annotated_image.png")
        annotated_img_path = os.path.join(self.base_folder, "annotated_image.png")
        # Save the cropped image
        cropped_image.save(annotated_img_path)  # Update the path as needed

        messagebox.showinfo(
            "Success", f"Annotated image saved successfully at {annotated_img_path}"
        )

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    model_path = "/Users/satyampant/Desktop/Uni/best_model.h5"
    base_folder = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/test"
    app = ImageClassifierApp(model_path, base_folder)
    app.run()

# %%
