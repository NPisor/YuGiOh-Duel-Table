import os
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

CARD_IMAGE_FOLDER = "card_art"
BOX_FOLDER = "tesseract_training_output"

class CardBoxViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Card Box Viewer")

        # Load card images and box files
        self.cards = [f for f in os.listdir(CARD_IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))]
        self.cards.sort()  # Sort alphabetically
        self.current_index = 0

        self.image_label = Label(self.master)
        self.image_label.pack()

        self.master.bind("<Left>", self.prev_card)
        self.master.bind("<Right>", self.next_card)

        self.show_card()

    def load_box_file(self, box_path):
        """Loads box file and returns a list of box data."""
        boxes = []
        try:
            with open(box_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 6:
                        char, left, bottom, right, top, _ = parts
                        boxes.append((char, int(left), int(bottom), int(right), int(top)))
        except FileNotFoundError:
            print(f"Box file not found: {box_path}")
        return boxes

    def show_card(self):
        """Displays the current card image with bounding boxes."""
        card_name = self.cards[self.current_index]
        card_path = os.path.join(CARD_IMAGE_FOLDER, card_name)
        box_path = os.path.join(BOX_FOLDER, os.path.splitext(card_name)[0] + ".box")

        # Load the image
        image = cv2.imread(card_path)
        if image is None:
            print(f"Image not found: {card_path}")
            return

        # Load and draw boxes
        boxes = self.load_box_file(box_path)
        for char, left, bottom, right, top in boxes:
            cv2.rectangle(image, (left, image.shape[0] - top), (right, image.shape[0] - bottom), (0, 255, 0), 2)
            cv2.putText(image, char, (left, image.shape[0] - bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Convert to Tkinter-compatible image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)

        # Update label
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def prev_card(self, event):
        """Go to the previous card."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_card()

    def next_card(self, event):
        """Go to the next card."""
        if self.current_index < len(self.cards) - 1:
            self.current_index += 1
            self.show_card()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardBoxViewer(root)
    root.mainloop()
