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


class ImageClassifierApp:
    def __init__(self, model_path, base_folder):
        self.model = self.load_pretrained_model(model_path)
        self.base_folder = base_folder
        self.window = tk.Tk()
        self.window.title("Image Classifier with LIME Explanation")
        self.target_size = (224, 224)
        self.draw_color = "green"
        self.eraser_on = False
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

        self.augment_button = tk.Button(
            self.window, text="Augment Image", command=self.augment_image
        )
        self.augment_button.pack()

        self.canvas = tk.Canvas(self.window, width=224, height=224)
        self.canvas.pack()

        self.toggle_eraser_button = tk.Button(
            self.window, text="Toggle Eraser", command=self.toggle_eraser
        )
        self.toggle_eraser_button.pack()

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
            self.toggle_eraser_button.pack()  # Show the eraser toggle button

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
        if self.eraser_on:
            # Erase by drawing transparent rectangles
            self.canvas.create_rectangle(
                x - size,
                y - size,
                x + size,
                y + size,
                outline="",
                fill="",
                stipple="gray12",
            )

        else:
            # Draw normally
            self.canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                fill=self.draw_color,
                outline=self.draw_color,
            )

    def toggle_eraser(self):
        self.eraser_on = not self.eraser_on
        self.draw_color = "gray12" if self.eraser_on else "red"

    def save_annotated_image(self):
        # # Ensure the canvas is at the top left of the window
        # self.window.update()
        # x = self.window.winfo_rootx() + self.canvas.winfo_x()
        # y = self.window.winfo_rooty() + self.canvas.winfo_y()
        # x1 = x + self.canvas.winfo_width()
        # y1 = y + self.canvas.winfo_height()
        # ImageGrab.grab().crop((x, y, x1, y1)).save("annotated_image.png")
        # Save the current canvas content as a PostScript file

        ps_filename = "canvas_output.ps"
        self.canvas.postscript(file=ps_filename, colormode="color")

        # Use PIL to convert the PostScript file to PNG
        try:
            with Image.open(ps_filename) as img:
                img.save(
                    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/annotated_img.png"
                )
            os.remove(ps_filename)  # To remove the PostScript file
            messagebox.showinfo("Success", "Image saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image : {e}")
    def augment_image(self):
        if hasattr(
            self, "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/annotated_img.png"
        ) and os.path.exists(self.uncertain_image_path):
            img = Image.open(self.uncertain_image_path)
            img = img.resize(self.target_size, Image.ANTIALIAS)
            img_array = np.array(img) / 255.0  # Normalize the image array for skimage

            # Perform augmentations
            scaled_img = self.scale_image(img_array)
            rotated_img = self.rotate_image(scaled_img)
            translated_img = self.translate_image(rotated_img)

            # Convert back to PIL Image and display or save
            final_img = Image.fromarray((translated_img * 255).astype(np.uint8))
            self.display_final_augmented_image(
                final_img
            )  # Implement this method to display the image
            final_img.save(
                "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augmented.png"
            )  # Save the final augmented image
        else:
            messagebox.showerror("Error", "No image loaded or image path is incorrect")
    
    def display_final_augmented_image(self, image):
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    model_path = "/Users/satyampant/Desktop/Uni/best_model.h5"
    base_folder = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/test"
    app = ImageClassifierApp(model_path, base_folder)
    app.run()


# ###### OLD CAIPI code
# #
# import tkinter as tk
# from tkinter import filedialog, messagebox, simpledialog
# from PIL import Image, ImageTk, ImageGrab
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.resnet50 import preprocess_input
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import os


# class ImageClassifierApp:
#     def __init__(self, model_path, base_folder):
#         self.model = self.load_pretrained_model(model_path)
#         self.base_folder = base_folder
#         self.window = tk.Tk()
#         self.window.title("Image Classifier with LIME Explanation")
#         self.target_size = (224, 224)
#         self.draw_color = "green"
#         self.eraser_on = False
#         self.orignal_image = None
#         self.setup_ui()
#         self.annotation_bounds = [
#             float("inf"),
#             float("inf"),
#             -float("inf"),
#             -float("inf"),
#         ]  # minX, minY, maxX, maxY

#     def load_pretrained_model(self, model_path):
#         try:
#             return load_model(model_path)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return None

#     def setup_ui(self):
#         self.load_button = tk.Button(
#             self.window, text="Load Image", command=self.load_image
#         )
#         self.load_button.pack()

#         self.predict_button = tk.Button(
#             self.window, text="Predict Image", command=self.predict_image_and_explain
#         )
#         self.predict_button.pack()

#         self.confirm_prediction_button = tk.Button(
#             self.window, text="Confirm Prediction", command=self.confirm_prediction
#         )
#         self.confirm_prediction_button.pack()

#         self.confirm_explanation_button = tk.Button(
#             self.window, text="Confirm Explanation", command=self.confirm_explanation
#         )
#         self.confirm_explanation_button.pack()

#         self.image_label = tk.Label(self.window)
#         self.image_label.pack()

#         self.prediction_label = tk.Label(self.window)
#         self.prediction_label.pack()

#         self.canvas = tk.Canvas(self.window, width=224, height=224)
#         self.canvas.pack()

#         self.toggle_eraser_button = tk.Button(
#             self.window, text="Toggle Eraser", command=self.toggle_eraser
#         )
#         self.toggle_eraser_button.pack()

#         self.save_button = tk.Button(
#             self.window, text="Save Annotated Image", command=self.save_annotated_image
#         )
#         self.save_button.pack()

#     def load_image(self):
#         file_path = filedialog.askopenfilename()
#         if file_path:
#             self.uncertain_image_path = file_path
#             img = Image.open(file_path)
#             img = img.resize(self.target_size, Image.ANTIALIAS)
#             photo = ImageTk.PhotoImage(img)
#             self.image_label.config(image=photo)
#             self.image_label.image = photo
#             self.canvas.bind("<B1-Motion>", self.draw_on_image)
#             self.annotation_bounds = [
#                 float("inf"),
#                 float("inf"),
#                 -float("inf"),
#                 -float("inf"),
#             ]

#     def predict_image_and_explain(self):
#         if hasattr(self, "uncertain_image_path") and os.path.exists(
#             self.uncertain_image_path
#         ):
#             predicted_label, prediction = self.predict_image(self.uncertain_image_path)
#             self.prediction_label.config(
#                 text=f"Prediction: {predicted_label}, Confidence: {np.max(prediction)}"
#             )

#             explanation_image = self.explain_prediction(self.uncertain_image_path)
#             self.display_explanation_on_canvas(explanation_image)
#         else:
#             self.prediction_label.config(
#                 text="No image loaded or image path is incorrect"
#             )

#     def predict_image(self, image_path):
#         img = load_img(image_path, target_size=self.target_size)
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         prediction = self.model.predict(img_array)
#         labels = ["bad", "good"]
#         predicted_label = labels[np.argmax(prediction)]
#         return predicted_label, prediction

#     def explain_prediction(self, image_path):
#         img = load_img(image_path, target_size=self.target_size)
#         img_array = img_to_array(img)
#         explainer = lime_image.LimeImageExplainer()
#         explanation = explainer.explain_instance(
#             img_array.astype("double"),
#             self.model.predict,
#             top_labels=2,
#             hide_color=0,
#             num_samples=1000,
#             num_features=10,
#         )
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=False,
#             num_features=10,
#             hide_rest=False,
#         )
#         explanation_image = mark_boundaries(temp / 2 + 0.5, mask)
#         explanation_image = Image.fromarray((explanation_image * 255).astype("uint8"))
#         return explanation_image

#     def display_explanation_on_canvas(self, explanation_image):
#         self.explained_photo = ImageTk.PhotoImage(explanation_image)
#         self.canvas.create_image(0, 0, anchor="nw", image=self.explained_photo)
#         self.canvas.image = self.explained_photo  # Keep reference

#     def confirm_prediction(self):
#         response = messagebox.askyesno(
#             "Confirm Prediction", "Is the prediction correct?"
#         )
#         if response:
#             messagebox.showinfo(
#                 "Prediction Confirmation", "Prediction confirmed as correct."
#             )
#         else:
#             self.change_prediction_label()

#     def change_prediction_label(self):
#         new_label = simpledialog.askstring("Input", "Enter the correct label:")
#         if new_label:
#             self.corrected_label = new_label
#             self.prediction_label.config(text=f"Corrected Label: {new_label}")

#     def confirm_explanation(self):
#         response = messagebox.askyesno(
#             "Confirm Explanation", "Is the explanation correct?"
#         )
#         if not response:
#             self.canvas.bind("<B1-Motion>", self.draw_on_image)
#             # self.canvas.bind("<B3-Motion>", self.erase_on_image)

#     def draw_on_image(self, event):
#         x, y = event.x, event.y
#         size = 10 if self.eraser_on else 5
#         self.canvas.create_oval(
#             x - size,
#             y - size,
#             x + size,
#             y + size,
#             fill=self.draw_color,
#             outline=self.draw_color,
#         )
#         # Update annotation bounds
#         self.annotation_bounds[0] = min(self.annotation_bounds[0], x - size)
#         self.annotation_bounds[1] = min(self.annotation_bounds[1], y - size)
#         self.annotation_bounds[2] = max(self.annotation_bounds[2], x + size)
#         self.annotation_bounds[3] = max(self.annotation_bounds[3], y + size)

#     def toggle_eraser(self):
#         self.eraser_on = not self.eraser_on
#         self.draw_color = "white" if self.eraser_on else "red"

#     def save_annotated_image(self):
#         if self.original_image and all(
#             b != float("inf") for b in self.annotation_bounds
#         ):
#             annotated_part = self.original_image.crop(self.annotation_bounds)
#             annotated_part.save(
#                 "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/annotated_image.png"
#             )
#             messagebox.showinfo("Save", "Annotated image saved.")
#         else:
#             messagebox.showinfo("Save", "No annotations made or image not loaded.")

#     def run(self):
#         self.window.mainloop()


# if __name__ == "__main__":
#     model_path = "/Users/satyampant/Desktop/Uni/best_model.h5"
#     base_folder = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/test"
#     app = ImageClassifierApp(model_path, base_folder)
#     app.run()

# #

# %%
