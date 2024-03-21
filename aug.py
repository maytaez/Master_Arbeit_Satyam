# %%
import os
import cv2
import albumentations as al


class ImageAugmentor:
    def __init__(self, input_dir, output_dir, height=224, width=224):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.height = height
        self.width = width
        self.transform = al.Compose(
            [
                al.Resize(width=self.width, height=self.height, p=1.0),
                al.Rotate(limit=90, p=1.0),
                al.HorizontalFlip(p=0.3),
                al.VerticalFlip(p=0.2),
                al.ColorJitter(contrast=2, p=0.2),
                al.ColorJitter(brightness=2, p=0.2),
                al.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
            ]
        )

    def augment_images(self, num_augmentations):
        print(f"Augmenting images in {self.input_dir}")
        for infile in os.listdir(self.input_dir):
            image = cv2.imread(os.path.join(self.input_dir, infile))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for j in range(num_augmentations):
                transformed = self.transform(image=image)
                transformed_image = transformed["image"]
                transformed_image = transformed_image * 255
                transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
                fname = os.path.join(self.output_dir, f"{infile[:-4]}_{j}.jpg")
                cv2.imwrite(fname, transformed_image)
        print("Augmentation finished.")


# Directory paths
dataset_url = "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/weld"
train_good_dir = os.path.join(dataset_url, "train", "good")
train_bad_dir = os.path.join(dataset_url, "train", "bad")
augmented_train_good_dir = (
    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/aug_train/good"
)
augmented_train_bad_dir = (
    "/Users/satyampant/Desktop/Uni/Master_Arbeit_Satyam/augment/aug_train/bad"
)

# Create instances of ImageAugmentor and augment images
augmentor_good = ImageAugmentor(train_good_dir, augmented_train_good_dir)
augmentor_bad = ImageAugmentor(train_bad_dir, augmented_train_bad_dir)

# Augment images
augmentor_good.augment_images(num_augmentations=20)
augmentor_bad.augment_images(num_augmentations=125)

# %%
