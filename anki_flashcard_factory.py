import requests
from utils import *
ANKI_CONNECT_URL = get_param_value("madzia_config.txt", "ANKI_CONNECT_URL")


def invoke(action, params=None):
    return requests.post(ANKI_CONNECT_URL, json={
        "action": action,
        "version": 6,
        "params": params or {}
    }).json()


def add_words_to_deck(deck_name, flashcards):
    notes = []
    for card in flashcards:
        notes.append({
            "deckName": deck_name,
            "modelName": "Basic",
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


def create_deck(deck_name):
    create_deck_response = invoke("createDeck", {"deck": deck_name})
    print("Create Deck:", create_deck_response)
