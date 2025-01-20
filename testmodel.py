from ultralytics import YOLO
import cv2
from fuzzywuzzy import fuzz
import easyocr
import sqlite3
import os
from tkinter import Tk, Label
from PIL import Image, ImageTk

# Paths
MODEL_PATH = "training_results/card_detection4/weights/best.pt"
CARD_ART_FOLDER = "card_art"
DATABASE_PATH = "card_list.db"
TABLE_NAME = "card_list"
COLUMN_NAME = "name"

# Load YOLO and EasyOCR models
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=True)

# Tracked cards
tracked_cards = {}

# Tkinter initialization
root = Tk()  # Initialize the Tkinter root window
root.title("Card Detection Info")
card_image_label = Label(root)
card_image_label.pack()
info_label = Label(root, text="", font=("Helvetica", 14), justify="left", wraplength=300)
info_label.pack()

# Keep track of whether the window is showing
window_active = False


def update_info_window(card_name, card_info):
    """
    Update the Tkinter window with the card's image and database information.
    Dynamically scales the window to fit the content.
    Args:
        card_name (str): Name of the detected card.
        card_info (str): Additional information about the card from the database.
    """
    global window_active

    # If no cards detected, hide the window
    if not card_name:
        if window_active:
            root.withdraw()  # Hide the window
            window_active = False
        return

    try:
        # Show the window if hidden
        if not window_active:
            root.deiconify()  # Show the window
            window_active = True

        # Default dimensions for the card image
        target_width, target_height = 150, 225

        card_name = "".join(c for c in card_name if c.isalnum() or c.isspace() or c == "-").lower()

        # Load and display the card image
        card_image_path = os.path.join(CARD_ART_FOLDER, f"{card_name}.jpg")
        if not os.path.exists(card_image_path):
            card_image_path = os.path.join(CARD_ART_FOLDER, f"{card_name}.png")  # Try .png if .jpg doesn't exist

        if os.path.exists(card_image_path):
            pil_image = Image.open(card_image_path)

            # Resize the image to fit a smaller window
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            card_image_label.configure(image=tk_image)
            card_image_label.image = tk_image
        else:
            card_image_label.configure(image=None)
            card_image_label.image = None

        # Wrap the text and display it in a compact format
        info_label.configure(text=card_info, font=("Helvetica", 10), wraplength=300)  # Set wraplength to 300 pixels
        root.geometry(f"{max(target_width + 50, 350)}x{target_height + 200}")  # Adjust window size dynamically
        root.update()
    except Exception as e:
        print(f"Error updating info window: {e}")


def detect_cards_live():
    global tracked_cards

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam for card detection. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.predict(source=frame, imgsz=640, conf=0.5, iou=0.45)

        current_tracked_ids = set()  # IDs of cards detected in the current frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]

                for card_id, card_info in tracked_cards.items():
                    if iou((x1, y1, x2, y2), card_info["bbox"]) > 0.5:
                        tracked_cards[card_id]["bbox"] = (x1, y1, x2, y2)
                        current_tracked_ids.add(card_id)
                        break
                else:
                    if conf > 0.6:
                        cropped_image = frame[y1:y2, x1:x2]
                        ocr_results = reader.readtext(cropped_image)
                        ocr_text = " ".join([res[1] for res in ocr_results])
                        matched_card, match_confidence, card_info = query_database(ocr_text)

                        if match_confidence > 75:
                            print(f"Matched Card: {matched_card} (Confidence: {match_confidence}%)")
                            card_id = len(tracked_cards)
                            tracked_cards[card_id] = {
                                "name": matched_card,
                                "bbox": (x1, y1, x2, y2),
                                "frames_lost": 0,
                            }
                            current_tracked_ids.add(card_id)
                            update_info_window(matched_card, f"Name: {matched_card}\nDetails: {card_info}")
                        else:
                            print(f"Unmatched Card: {ocr_text}")

        # Forget cards that are no longer in the frame
        for card_id in list(tracked_cards.keys()):
            if card_id not in current_tracked_ids:
                tracked_cards[card_id]["frames_lost"] += 1
                if tracked_cards[card_id]["frames_lost"] > 5:  # Adjust threshold as needed
                    del tracked_cards[card_id]

        # Close the Tkinter window if no cards are detected
        if not tracked_cards:
            update_info_window(None, None)

        # Draw bounding boxes and labels for tracked cards
        for card_id, card_info in tracked_cards.items():
            x1, y1, x2, y2 = card_info["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, card_info["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Card Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def query_database(card_text):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute(f"SELECT {COLUMN_NAME} FROM {TABLE_NAME}")
    card_names = cursor.fetchall()
    cursor.execute(f"SELECT * FROM {TABLE_NAME}")
    all_card_info = cursor.fetchall()
    connection.close()

    best_match, best_confidence, best_card_info = None, 0, None
    for card_name, card_info in zip(card_names, all_card_info):
        card_name = card_name[0]
        confidence = fuzz.ratio(card_text.lower(), card_name.lower())
        if confidence > best_confidence:
            best_match, best_confidence, best_card_info = card_name, confidence, card_info
    return best_match, best_confidence, best_card_info


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


if __name__ == "__main__":
    detect_cards_live()
