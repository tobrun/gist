#!/usr/bin/env python3
"""
Simple script to demonstrate the usage of Gemma-3 for image description
"""

# Specific version of transformers needed when Gemma-3 was released.
# pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import time

# Debug torch setup
print(f"Is CUDA Available: {torch.cuda.is_available()}")
print(f"Amount of detected GPUs: {torch.cuda.device_count()}")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "../models/gemma-3-27b-it"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load model
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)
prompt = "<start_of_image> in this image, there is"

# Process inputs and ensure they're on the right device
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
input_len = model_inputs["input_ids"].shape[-1]

# Time the generation and decoding
generation_start_time = time.time()
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False
    )    
    generation = generation[0][input_len:]
generation_end_time = time.time()
generation_time = generation_end_time - generation_start_time

# Time the generation
start_time = time.time()
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False
    )    
    generation = generation[0][input_len:]
end_time = time.time()
generation_time = end_time - start_time

decoded = processor.decode(generation, skip_special_tokens=True)
print(prompt + decoded)
print(f"\nGeneration time: {generation_time:.2f} seconds")