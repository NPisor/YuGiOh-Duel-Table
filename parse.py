import json
import os
import requests

# Define the folder to save images
os.makedirs("card_art", exist_ok=True)

# Open and parse the JSON file with utf-8 encoding
with open("yugioh_dump.json", "r", encoding="utf-8") as file:
    card_data = json.load(file)

# Base URL for the card images
base_image_url = "https://images.ygoprodeck.com/images/cards_small/"

# Iterate over each card and download its image
for card in card_data:  # Iterate over the list of cards directly
    card_id = card["id"]
    card_name = card["name"]

    # Construct image URL
    image_url = f"{base_image_url}{card_id}.jpg"

    # Sanitize card name for the filename
    sanitized_name = "".join(c for c in card_name if c.isalnum() or c in (" ", "-")).strip()
    file_name = f"card_art/{sanitized_name}.jpg"

    try:
        # Fetch the image
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Save the image
            with open(file_name, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Downloaded: {sanitized_name}")
        else:
            print(f"Failed to download {sanitized_name}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {sanitized_name}: {e}")

print("Image download complete.")
