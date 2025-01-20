import requests
import json

# Define base URL and headers
url = "https://ygoprodeck.com/api/elastic/card_search.php"

headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en-US,en;q=0.9",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://ygoprodeck.com/card-database/?num=100&offset=0",
    "cookie": "your_cookies_here",  # Replace with your actual cookies if necessary
}

# Parameters for pagination
params = {
    "num": 100,  # Number of results per page
    "offset": 0,  # Starting offset
}

all_cards = []  # List to store all cards

while True:
    print(f"Fetching cards with offset {params['offset']}...")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        try:
            data = response.json()  # Parse JSON response
            if "cards" in data and data["cards"]:  # Check if cards are present
                all_cards.extend(data["cards"])  # Append to the list
                print(f"Fetched {len(data['cards'])} cards. Total: {len(all_cards)}")
                
                # Update offset for next page
                params["offset"] += 100
                
                # Check if there are no more cards
                if "paging" in data and data["paging"].get("rows_remaining", 0) == 0:
                    break
            else:
                print("No more cards found.")
                break
        except json.JSONDecodeError:
            print("Failed to decode JSON. Check the response content:")
            print(response.text)
            break
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        break

# Save all cards to a local JSON file
output_file = "yugioh_dump.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(all_cards, file, ensure_ascii=False, indent=4)

print(f"All card data saved to {output_file}")
