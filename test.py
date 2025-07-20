from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a small model with no login needed
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

system_message = '''
You are an intelligent language assistant. The user will give you a list of words extracted via OCR from an English book with sophisticated language. These words may contain typos or OCR errors.

Your task is to:

Correct each word if it contains spelling errors, using literary vocabulary as context.

Return a list of (word, definition) pairs, where:

The word is the corrected form.

The definition is clear, concise, and appropriate for an educated reader.

If a word is severely corrupted or ambiguous, guess the likely intended word and annotate it like:
[possibly intended: '____']. Provide only one definition per word, once and in order.

Format your answer as:

1. word — definition '\n'


It is very important that this format is kept. Revise the output before outputting.


'''

words = ['estranged',
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
         'sullenly']
prompt = f"<|system|>\n{system_message}\n<|user|>\nThis is the list of words {words}\n<|assistant|>\n"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
