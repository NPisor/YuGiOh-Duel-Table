import os
import csv

IMAGE_FOLDER = "card_art"  # Path to your image folder
OUTPUT_CSV = "labels.csv"  # Output CSV file

def generate_labels_csv(image_folder, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])  # Header row

        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg"):  # Adjust for your image format
                label = os.path.splitext(filename)[0].replace("_", " ")
                writer.writerow([filename, label])

    print(f"Labels CSV generated: {output_csv}")

generate_labels_csv(IMAGE_FOLDER, OUTPUT_CSV)
