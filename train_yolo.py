import os
from ultralytics import YOLO
import yaml

# Paths
DATASET_FOLDER = "yolo_training_data"  # Output folder from the dataset preparation script
MODEL_CHECKPOINT = "yolov5s.pt"  # Pre-trained YOLO model (small version as a starting point)
CONFIG_FILE = "yolo_config.yaml"
OUTPUT_DIR = "training_results"

# Step 1: Create a YOLO config file
def create_config():
    config = {
        "path": DATASET_FOLDER,
        "train": os.path.join(DATASET_FOLDER, "images"),  # Training images
        "val": os.path.join(DATASET_FOLDER, "images"),    # Validation images (same as training for now)
        "names": ["card"],                                # Single class: card
        "nc": 1                                           # Number of classes
    }
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(config, file)
    print(f"Config file '{CONFIG_FILE}' created successfully.")

# Step 2: Train YOLO model
def train_yolo():
    model = YOLO(MODEL_CHECKPOINT)  # Load a pre-trained YOLO model (e.g., YOLOv5s)
    model.train(
        data=CONFIG_FILE,
        epochs=50,                  # Number of training epochs
        imgsz=640,                  # Image size
        batch=16,                   # Batch size
        project=OUTPUT_DIR,         # Directory to save results
        name="card_detection"       # Run name
    )

# Step 3: Validate the model (optional)
def validate_model():
    model = YOLO(os.path.join(OUTPUT_DIR, "card_detection8/weights/best.pt"))  # Load best model
    results = model.val()  # Validate the model
    print("Validation results:", results)

# Step 4: Perform inference on test images (optional)
def test_inference():
    model = YOLO(os.path.join(OUTPUT_DIR, "card_detection/weights/best.pt"))  # Load best model
    test_images = [
        "path/to/test_image1.jpg",
        "path/to/test_image2.jpg"
    ]  # Replace with actual paths
    for image_path in test_images:
        results = model.predict(image_path, save=True)  # Save annotated images
        print(f"Inference completed for {image_path}. Results saved.")

if __name__ == "__main__":

    # Step 2: Train the model
    #train_yolo()

    # (Optional) Step 3: Validate the model
    validate_model()

    # (Optional) Step 4: Test inference
    # test_inference()
