def get_key_from_file(file_name):
    with open(file_name, 'r') as file:
            line = file.readline()
    return line.strip()
api_key = get_key_from_file("mw_key.txt")


import requests


def get_definition(word, api_key):
    url = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")

    data = response.json()

    # Check if word was found
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        definitions = []
        for entry in data:
            if 'shortdef' in entry:
                definitions.extend(entry['shortdef'])
        return definitions[0]
    else:
        return [f"No definitions found. Suggestions: {data}"]



API_KEY = api_key

defs = get_definition("covet", API_KEY)

print(defs)
