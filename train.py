import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import csv

LABELS = {}
name_to_label = {}  # Map card names to numeric labels
current_label = 0   # Start assigning labels from 0

# Read the CSV file
with open("labels.csv", "r") as f:
    reader = csv.reader(f)
    # skip first row
    next(reader)
    for row in reader:
        filename, card_name = row
        filename = filename.strip()  # Remove extra spaces
        card_name = card_name.strip()

        # Assign a numeric label if the card name hasn't been seen before
        if card_name not in name_to_label:
            name_to_label[card_name] = current_label
            current_label += 1

        # Map the filename to its numeric label
        LABELS[filename] = name_to_label[card_name]

print(f"Parsed {len(LABELS)} labels.")

# Define dataset
class CardDataset(Dataset):
    def __init__(self, image_folder, labels, transform=None):
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform
        self.image_files = list(labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = self.labels[img_name]
        img_path = os.path.join(self.image_folder, img_name)

        # Debugging: Check if file exists
        if not os.path.exists(img_path):
            print(f"Missing image file: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Define model
def build_pytorch_model(num_classes):
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    return base_model


# Training loop
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Main script
if __name__ == "__main__":
    IMAGE_FOLDER = "card_art"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = len(set(LABELS.values()))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Verify only listed files exist
    missing_files = [
        img_name for img_name in LABELS if not os.path.exists(os.path.join(IMAGE_FOLDER, img_name))
    ]
    if missing_files:
        print(f"Missing {len(missing_files)} files:")
        for missing_file in missing_files:
            print(f"  {missing_file}")
        print("Ensure all listed files exist in the card_art folder.")
        exit(1)

    print(f"All {len(LABELS)} listed files are present.")

    # Prepare dataset and dataloader
    dataset = CardDataset(IMAGE_FOLDER, LABELS, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    model = build_pytorch_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Starting training...")
    try:
        train_model(model, dataloader, criterion, optimizer, DEVICE, num_epochs=10)
        print("Training completed!")
    except FileNotFoundError as e:
        print(e)
