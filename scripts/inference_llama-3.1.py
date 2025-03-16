#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference script for Llama 3.1 fine-tuned model')
    parser.add_argument('--model_path', type=str, default="./", 
                        help='Path to the model directory')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                        help='Repetition penalty')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"Loading model from: {args.model_path}")
    print(f"Using device: {args.device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map=args.device
    )
    
    print("Model loaded successfully!")
    
    # Interactive loop
    while True:
        user_input = input("\nEnter your prompt (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Format the prompt as per Llama 3.1 chat template
        prompt = f"<|system|>\nYou are a helpful assistant.<|end_of_turn|>\n<|user|>\n{user_input}<|end_of_turn|>\n<|assistant|>\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        # Setup streamer for real-time output
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate in a separate thread to allow streaming
        generation_kwargs = dict(
            inputs=inputs.input_ids,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            streamer=streamer,
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Print the output as it's generated
        print("\nAssistant: ", end="", flush=True)
        for text in streamer:
            print(text, end="", flush=True)
        print("\n")
        
        thread.join()

if __name__ == "__main__":
    main()