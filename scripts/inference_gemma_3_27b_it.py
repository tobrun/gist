#!/usr/bin/env python3
"""
Unified script to demonstrate the usage of Gemma-3 for text and/or image processing.

Usage:
  - Text only: python script.py --prompt "Your text prompt here"
  - Image only: python script.py --image_path path/to/image.jpg
  - Image from URL: python script.py --image_url "https://example.com/image.jpg"
  - Image with text: python script.py --image_path path/to/image.jpg --prompt "Describe this image"
"""

# Specific version of transformers needed when Gemma-3 was released.
# pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
# pip install accelerate

import argparse
import requests
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Gemma-3 model with text and/or image input")
    parser.add_argument("--model_path", type=str, default="../models/gemma-3-27b-it",
                        help="Path to the Gemma-3 model")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for the model")
    parser.add_argument("--image_path", type=str,
                        help="Path to local image file")
    parser.add_argument("--image_url", type=str,
                        help="URL to download image from")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")
    return parser.parse_args()

def load_model(model_path):
    """Load the Gemma-3 model and processor."""
    print("Loading model from:", model_path)

    # Debug torch setup
    print(f"Is CUDA Available: {torch.cuda.is_available()}")
    print(f"Amount of detected GPUs: {torch.cuda.device_count()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor, device

def load_image(args):
    """Load an image from a URL or local path if specified."""
    image = None

    if args.image_url:
        print(f"Loading image from URL: {args.image_url}")
        image = Image.open(requests.get(args.image_url, stream=True).raw)
    elif args.image_path:
        print(f"Loading image from path: {args.image_path}")
        image = Image.open(args.image_path)

    return image

def run_generation(model, processor, args, device):
    """Run generation with the model using text and/or image input."""
    image = load_image(args)
    prompt = args.prompt

    if not image and not prompt:
        raise ValueError("At least one of --prompt, --image_path, or --image_url must be provided")

    # Print input information
    if image:
        print(f"Using image input")
    if prompt:
        print(f"Using text prompt: {prompt}")

    # Process inputs and ensure they're on the right device
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_len = model_inputs["input_ids"].shape[-1]

    # Time the generation
    start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.do_sample
        )
        generation = generation[0][input_len:]
    end_time = time.time()
    generation_time = end_time - start_time

    decoded = processor.decode(generation, skip_special_tokens=True)

    # Print results
    if prompt:
        print(f"\nPrompt: {prompt}")
    print(f"Response: {decoded}")
    print(f"\nGeneration time: {generation_time:.2f} seconds")

    return decoded, generation_time

def main():
    """Main function to run the script."""
    args = parse_arguments()
    model, processor, device = load_model(args.model_path)
    run_generation(model, processor, args, device)

if __name__ == "__main__":
    main()