import cv2
import pytesseract
import sqlite3
from PIL import Image, ImageTk
from tkinter import Tk, Label, Toplevel
from pytesseract import Output
from fuzzywuzzy import process
import os
import numpy as np

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Database path
DB_PATH = "card_list.db"

def lookup_card_fuzzy(name):
    """
    Perform a fuzzy search in the database to find the closest matching card.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM card_list")
    all_cards = cursor.fetchall()
    conn.close()

    # Extract all card names from the database
    card_names = [card[1] for card in all_cards]  # Assuming card name is in the second column

    # Use fuzzy matching to find the closest match
    match = process.extractOne(name, card_names)
    if match and match[1] > 70:  # Only consider matches with confidence > 70
        for card in all_cards:
            if card[1].lower() == match[0].lower():
                return card
    return None

def display_card_info(card_data):
    """
    Display the card information and image in a popup window.
    """
    if card_data:
        card_id, name, text, card_type, race, attribute, atk, df, tcg_date, ocg_date, image_name = card_data
        top = Toplevel()
        top.title(name)

        # Display card image
        image_path = f'card_art/{image_name}'
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((200, 300))  # Resize for display
            photo = ImageTk.PhotoImage(img)
            img_label = Label(top, image=photo)
            img_label.image = photo
            img_label.pack()
        else:
            Label(top, text="Image not found").pack()

        # Display card information
        info = f"""
        Name: {name}
        Text: {text}
        Type: {card_type}
        Race: {race}
        Attribute: {attribute}
        ATK: {atk}
        DEF: {df}
        TCG Release: {tcg_date}
        OCG Release: {ocg_date}
        """
        info_label = Label(top, text=info, justify="left")
        info_label.pack()
    else:
        print("Card not found!")

def group_words(ocr_data):
    """Group detected words based on positions and combine into a single string."""
    grouped_text = []
    prev_top = -1

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        if word and conf > 50:
            top = ocr_data["top"][i]
            height = ocr_data["height"][i]

            if grouped_text and abs(top - prev_top) <= height + 5:
                grouped_text[-1] += f" {word}"
            else:
                grouped_text.append(word)
            prev_top = top

    return " ".join(grouped_text).strip()

def preprocess_image_color(image):
    """
    Preprocess the color image for better OCR accuracy using grayscale, resizing, unsharp masking, 
    and Canny edge detection.
    """
    image = cv2.imread(image)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image}")

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize the image (scale to 1191x2000)
    resized = cv2.resize(gray, (1191, 2000), interpolation=cv2.INTER_LINEAR)

    # Define ROI (upper part of the card for text)
    roi = resized[20:250, 20:1000]  # Adjust this region to focus on the desired area

    # Step 3: Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Step 4: Apply Unsharp Mask
    # Create a sharper version of the image
    unsharp = cv2.addWeighted(blurred, 2.69, cv2.GaussianBlur(blurred, (11, 11), 6.4), -1.69, 0)

    # Step 5: Apply Canny Edge Detection
    edges = cv2.Canny(unsharp, threshold1=190, threshold2=280)

    return edges



def process_webcam():
    """
    Capture video feed, perform OCR, and display card info.
    """
    cap = cv2.VideoCapture(0)  # Use the first webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define ROI (adjust as needed for your setup)
        roi = frame[80:320, 80:640]  # Adjust based on card size in the frame

        # Preprocess the ROI
        enhanced = preprocess_image_color(roi)

        # Perform OCR on the enhanced image
        ocr_data = pytesseract.image_to_data(enhanced, output_type=Output.DICT)

        # Group words into a single card name
        try:
            card_name = group_words(ocr_data)
        except IndexError:
            card_name = ""  # Handle cases where grouping fails

        # Clean up card name and search database
        if card_name:
            print(f"Detected card name: {card_name}")
            card_data = lookup_card_fuzzy(card_name)
            if card_data:
                display_card_info(card_data)
            else:
                print("No matching card found.")
        else:
            print("No valid text detected.")

        # Display the processed ROI and the text
        cv2.imshow("Enhanced ROI", enhanced)
        cv2.putText(frame, f"Detected: {card_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the webcam feed
        cv2.imshow("Webcam", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def perform_ocr(image_path):
    """
    Perform OCR on the preprocessed image and display the detected text.
    """
    preprocessed = preprocess_image_color(image_path)

    text = pytesseract.image_to_string(preprocessed, config="--psm 6")
    print("Detected Text:", text)

    # Display the preprocessed image
    cv2.imshow("Preprocessed Image", preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text.strip()


image_path = "results/sr_7 Colored Fish.png"

if __name__ == "__main__":

    perform_ocr(image_path)
    #process_webcam()

    

