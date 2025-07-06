import requests

ANKI_CONNECT_URL = "http://localhost:8765"


def invoke(action, params=None):
    return requests.post(ANKI_CONNECT_URL, json={
        "action": action,
        "version": 6,
        "params": params or {}
    }).json()


# 1. Create the deck
deck_name = "We Need to Talk about Kevin"
create_deck_response = invoke("createDeck", {"deck": deck_name})
print("Create Deck:", create_deck_response)

# 2. Prepare flashcards
flashcards = [
    {"word": "to covet", "definition": "to want very badly"},
]

# 3. Add notes (flashcards)
notes = []
for card in flashcards:
    notes.append({
        "deckName": deck_name,
        "modelName": "Basic",  # Uses Anki's default 'Basic' note type
        "fields": {
            "Front": card["word"],
            "Back": card["definition"]
        },
        "options": {
            "allowDuplicate": False
        },
        "tags": ["auto_added"]
    })

add_notes_response = invoke("addNotes", {"notes": notes})
print("Add Notes:", add_notes_response)
