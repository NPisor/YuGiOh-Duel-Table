import cv2
import easyocr
import os
import matplotlib.pyplot as plt

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you want to use your GPU

# Path to your card art folder
CARD_ART_FOLDER = "card_art"

def isolate_roi(image):
    """
    Isolate the region of interest (ROI) in the image.
    """
    # Resize the image to a fixed size (optional, to maintain consistency)
    resized = cv2.resize(image, (1191, 2000), interpolation=cv2.INTER_LINEAR)
    
    # Define the ROI (upper part of the card for text)
    roi = resized[20:250, 20:1000]  # Adjust these coordinates as needed
    return roi

def process_images():
    """
    Process the first 10 images in the folder with EasyOCR.
    """
    files = [f for f in os.listdir(CARD_ART_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))][:10]
    
    for filename in files:
        image_path = os.path.join(CARD_ART_FOLDER, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        # Isolate the ROI
        roi = isolate_roi(image)

        # Perform OCR on the ROI
        results = reader.readtext(roi)

        print(f"Results for {filename}:")
        for (bbox, text, confidence) in results:
            print(f"Text: {text}, Confidence: {confidence:.2f}")

        # Display the ROI using matplotlib
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
        plt.title(f"ROI - {filename}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    process_images()
