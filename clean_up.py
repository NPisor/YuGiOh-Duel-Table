import os
import torch
from PIL import Image
from RealESRGAN import RealESRGAN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# Input and output directories
input_dir = 'inputs'
output_dir = 'results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get the first 5 images from the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))][:5]

# Process each image
for i, image_file in enumerate(image_files, start=1):
    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, f'sr_{os.path.splitext(image_file)[0]}.png')

    try:
        print(f"Processing {i}/{len(image_files)}: {input_path}")
        image = Image.open(input_path).convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

print("Processing complete.")
