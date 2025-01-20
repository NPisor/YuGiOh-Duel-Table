import os
import random
from glob import glob
from shutil import rmtree, copyfile
import cv2
import albumentations as A
import numpy as np

class CardDatasetPreparer:
    def __init__(self, card_art_folder, output_folder, num_samples=150, train_ratio=0.8):
        self.card_art_folder = card_art_folder
        self.output_folder = output_folder
        self.num_samples = num_samples
        self.train_ratio = train_ratio

        # Create output directories for train and val sets
        self.train_images_dir = os.path.join(self.output_folder, "images", "train")
        self.val_images_dir = os.path.join(self.output_folder, "images", "val")
        self.train_labels_dir = os.path.join(self.output_folder, "labels", "train")
        self.val_labels_dir = os.path.join(self.output_folder, "labels", "val")
        
        # Clean up and recreate directories
        rmtree(self.output_folder, ignore_errors=True)
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.val_images_dir, exist_ok=True)
        os.makedirs(self.train_labels_dir, exist_ok=True)
        os.makedirs(self.val_labels_dir, exist_ok=True)

        # Albumentations pipeline for augmentations
        self.augmenter = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.GaussNoise(p=0.1),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.Resize(640, 640),  # Resize images for YOLO input
        ])

    def prepare_dataset(self):
        # Get all images from the card_art folder
        card_images = glob(os.path.join(self.card_art_folder, "*.jpg")) + \
                      glob(os.path.join(self.card_art_folder, "*.png"))
        if len(card_images) < self.num_samples:
            raise ValueError(f"Not enough images in {self.card_art_folder}. Found {len(card_images)}.")

        # Randomly select images
        selected_images = random.sample(card_images, self.num_samples)

        # Split into train and val sets
        split_idx = int(len(selected_images) * self.train_ratio)
        train_images = selected_images[:split_idx]
        val_images = selected_images[split_idx:]

        # Process train and val sets
        self._process_images(train_images, self.train_images_dir, self.train_labels_dir, "train")
        self._process_images(val_images, self.val_images_dir, self.val_labels_dir, "val")

    def _process_images(self, images, images_dir, labels_dir, split_name):
        for img_idx, img_path in enumerate(images):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Get image dimensions
            h, w, _ = image.shape

            # Apply augmentations
            augmented = self.augmenter(image=image)
            aug_image = augmented["image"]

            # Save the augmented image
            base_name = f"{split_name}_card_{img_idx:03d}"
            image_save_path = os.path.join(images_dir, f"{base_name}.jpg")
            cv2.imwrite(image_save_path, aug_image)

            # Create bounding box annotation (full image bounds)
            bbox = [0, 0, w, h]  # [x_min, y_min, x_max, y_max]
            bbox_normalized = [
                0,  # Class ID (0 for all cards in this example)
                (bbox[0] + bbox[2]) / 2 / w,  # x_center (normalized)
                (bbox[1] + bbox[3]) / 2 / h,  # y_center (normalized)
                (bbox[2] - bbox[0]) / w,      # width (normalized)
                (bbox[3] - bbox[1]) / h       # height (normalized)
            ]

            # Save annotation in YOLO format
            label_save_path = os.path.join(labels_dir, f"{base_name}.txt")
            with open(label_save_path, "w") as f:
                f.write(" ".join(map(str, bbox_normalized)))

        print(f"{split_name.capitalize()} dataset prepared: {len(images)} images")

if __name__ == "__main__":
    # Paths
    CARD_ART_FOLDER = "card_art"
    OUTPUT_FOLDER = "yolo_training_data"

    # Create and prepare dataset with 150 samples
    preparer = CardDatasetPreparer(CARD_ART_FOLDER, OUTPUT_FOLDER, num_samples=150)
    preparer.prepare_dataset()
