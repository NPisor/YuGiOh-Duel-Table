import sqlite3
import json
import os

# Define database and JSON file paths
db_path = "card_list.db"
json_path = "yugioh_dump.json"

# Create or connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the cards table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS card_list (
    id INTEGER PRIMARY KEY,
    name TEXT,
    text TEXT,
    type TEXT,
    race TEXT,
    attribute TEXT,
    atk INTEGER,
    df INTEGER,
    tcg_date TEXT,
    ocg_date TEXT,
    image_name TEXT
);
''')

# Commit table creation
conn.commit()

# Load the JSON file
if not os.path.exists(json_path):
    print(f"Error: JSON file '{json_path}' not found.")
    conn.close()
    exit(1)

with open(json_path, "r", encoding="utf-8") as file:
    card_data = json.load(file)

# Parse the JSON and populate the database
for card in card_data:
    card_id = card.get("id")
    name = card.get("name")
    text = card.get("desc")  # 'desc' in JSON maps to 'text'
    card_type = card.get("type")
    race = card.get("race")
    attribute = card.get("attribute")
    atk = card.get("atk", 0)  # Default to 0 if not present
    df = card.get("def", 0)   # Default to 0 if not present
    tcg_date = card.get("card_sets", [{}])[0].get("tcg_date", None)  # Handle potential nested structure
    ocg_date = card.get("card_sets", [{}])[0].get("ocg_date", None)  # Handle potential nested structure
    image_name = f"{name}.jpg" if name else None

    # Insert or update the card in the database
    cursor.execute('''
    INSERT INTO card_list (id, name, text, type, race, attribute, atk, df, tcg_date, ocg_date, image_name)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        name=excluded.name,
        text=excluded.text,
        type=excluded.type,
        race=excluded.race,
        attribute=excluded.attribute,
        atk=excluded.atk,
        df=excluded.df,
        tcg_date=excluded.tcg_date,
        ocg_date=excluded.ocg_date,
        image_name=excluded.image_name;
    ''', (card_id, name, text, card_type, race, attribute, atk, df, tcg_date, ocg_date, image_name))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database update complete.")
