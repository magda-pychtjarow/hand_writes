# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate
#
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
#
#
# mistral_tokenizer = MistralTokenizer.from_file("mistral-7B-v0.3/tokenizer.model.v3")
# completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
# tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
# model = Transformer.from_folder("mistral-7B-v0.3")
# out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)
# result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
# print(result)

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

start_time = time.time()
# Load Mistral from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Format the prompt with ChatML tags


# Prompt
system_msg = """You are an intelligent language assistant. The user will give you a list of words extracted via OCR from an English book with sophisticated language. 
These words may contain typos or OCR errors.
Your task is to:
Correct each word if it contains spelling errors, using literary vocabulary as context.
Return a list of words taking into consideration the corrected versions. If a word was corrected add a tag "CORRECTED" before the word.
 Only add it if you suggest a changed word.
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

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate output
outputs = model.generate(input_ids, max_new_tokens=500, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

print("Took ", str(time.time() - start_time / 60), " minutes")
