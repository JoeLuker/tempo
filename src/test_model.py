#!/usr/bin/env python3
import torch
from model_loader import load_model

def main():
    """Test basic model loading and generation"""
    print("Testing Mistral-7B model loading...")
    model, tokenizer = load_model()
    
    prompt = "In a surprising turn of events, scientists discovered that"
    print(f"\nGenerating completion for prompt: '{prompt}'")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    print("\nModel loading and generation test completed successfully!")

if __name__ == "__main__":
    main() 