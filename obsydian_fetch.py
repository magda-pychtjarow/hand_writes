from utils import *
import os
from anki_flashcard_factory import *

OBSIDIAN_VAULT_PATH = get_param_value("madzia_config.txt", "OBSIDIAN_VAULT_PATH")
print(list(os.walk(OBSIDIAN_VAULT_PATH)))
def get_obsydian_notes_paths():
    notes_paths = []
    for root, folders, files in os.walk(OBSIDIAN_VAULT_PATH):
        for file in files:
            if file.endswith(".md"):
                notes_paths.append(file)
        return notes_paths

print(get_obsydian_notes_paths())

def parse_words(nb_path):
    words = []
    with open(nb_path, "r") as f:
        text = f.read()

        words = text.split()  # split on any whitespace
        for word in words:
            words.extend(word)

    return words




def preprocess_obsydian_definition_notes(fetched_lines):
    flashcards = []
    for f_line in fetched_lines:
        f_line = f_line.strip()
        if " - " in f_line:
            word, definition = f_line.split(" - ", 1)
            flashcards.append({
                "word": word.strip(),
                "definition": definition.strip()
            })
    return flashcards




def get_flashcards(notebook_name):
    notes_files = get_obsydian_notes_paths()
    nb_name = [n for n in notes_files if n.replace(".md","") == notebook_name]

    if len(nb_name) == 0 :
        print(f"Missing notebook of name {nb_name}")
        return
    elif len(nb_name) > 1 :
        print("Incorrectly fetched.")
        return
    nb_name = nb_name[0]
    print(nb_name)
    words = []
    path = os.path.join(OBSIDIAN_VAULT_PATH, nb_name)
    with open(path) as f:
        for line in f:
            line = line.strip()
            words.append(line)
    return preprocess_obsydian_definition_notes(words)




deck_name = "De kleuren van anna"
create_deck(deck_name)
add_words_to_deck(deck_name, get_flashcards(deck_name))