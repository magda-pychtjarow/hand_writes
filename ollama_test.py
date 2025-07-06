import ollama

system_msg = """You are an intelligent language assistant. The user will give you a list of words extracted via OCR from an English book with sophisticated language. 
These words may contain typos or OCR errors.
Your task is to:
Correct each word if it contains spelling errors, using literary vocabulary as context.
Return a list of words taking into consideration the corrected versions. The list must begin by '[' and end with ']'. Each word should be enclosed in quotes, every word in single quotes. After each word, a new line.
"""

user_msg = '''['estranged',
 'tricksy',
 'chided',
 'telltale',
 'bhubles',
 'furtive',
 'ramrod',
 'svelte',
 'reel',
 'girded',
 'mitley',
 'albumen',
 'precarious',
 'pisgorged', 
 'intimations',
 'ablutions',
 'inchoate',
 'sullenly']'''

prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

response = ollama.generate(model='llama3.1:8b',
                           prompt=prompt)['response']

import re
import pandas as pd

entries = []
print(response)
lines = response.strip().split('\n')
for line in lines:
    matches = re.findall(r"'([^']+)'", line)
    if matches:
        entries.append(matches)

# Step 2: Create a DataFrame (table)
df = pd.DataFrame(entries)

# Print the table
print(df)
