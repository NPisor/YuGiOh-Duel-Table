import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Database and card art folder paths
db_path = "card_list.db"
art_folder = "card_art"

def search_card():
    card_name = entry.get()
    if not card_name:
        messagebox.showerror("Input Error", "Please enter a card name.")
        return

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the database
        cursor.execute("SELECT * FROM card_list WHERE name = ?", (card_name,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            messagebox.showinfo("Not Found", f"Card '{card_name}' not found in the database.")
            return

        # Extract card data
        card_id, name, text, card_type, race, attribute, atk, df, tcg_date, ocg_date, image_name = result

        # Display card information
        info_text = f"Name: {name}\nType: {card_type}\nRace: {race}\nAttribute: {attribute}\nATK: {atk}\nDEF: {df}\nTCG Date: {tcg_date}\nOCG Date: {ocg_date}\nText: {text}"
        info_box.delete("1.0", tk.END)
        info_box.insert(tk.END, info_text)

        # Load and display card image
        image_path = os.path.join(art_folder, image_name)
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((150, 200))  # Resize for display
            card_img = ImageTk.PhotoImage(img)

            # Update the label with the image
            card_image_label.config(image=card_img)
            card_image_label.image = card_img
        else:
            messagebox.showerror("Image Error", f"Image '{image_name}' not found.")

    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"An error occurred: {e}")

# Create the GUI window
window = tk.Tk()
window.title("Yu-Gi-Oh! Card Lookup")
window.geometry("400x600")

# Input box and search button
entry_label = tk.Label(window, text="Enter Card Name:")
entry_label.pack(pady=5)

entry = tk.Entry(window, width=30)
entry.pack(pady=5)

search_button = tk.Button(window, text="Search", command=search_card)
search_button.pack(pady=10)

# Image display
card_image_label = tk.Label(window)
card_image_label.pack(pady=10)

# Information display
info_box = tk.Text(window, height=15, width=50, wrap=tk.WORD)
info_box.pack(pady=10)

# Run the GUI loop
window.mainloop()
