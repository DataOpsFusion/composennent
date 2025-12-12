"""
Example: Loading a model and generating text.

This demonstrates the unified 'generate' API that works across
all model types in the library.
"""

import torch
from composennent.nlp.transformers import GPT, BaseLanguageModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    # In a real app: model = GPT.load("path/to/model.pt", device=device)
    print("Initializing model...")
    model = GPT(vocab_size=50257, latent_dim=768, num_heads=12, num_layers=12)
    model.to(device)
    model.eval()

    # 2. Prepare Input
    # Mock input IDs (normally from tokenizer.encode("Prompt"))
    prompt_ids = torch.randint(0, 50257, (1, 5)).to(device)
    
    print(f"Input shape: {prompt_ids.shape}")

    # 3. Generate
    # The API supports various decoding strategies out of the box
    
    print("\n--- Greedy Decoding ---")
    output = model.generate_greedy(
        prompt_ids, 
        max_length=20
    )
    print(f"Output shape: {output.shape}")

    print("\n--- Nucleus Sampling (Top-p) ---")
    output_sampled = model.generate(
        prompt_ids,
        max_length=30,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    print(f"Output shape: {output_sampled.shape}")

    # 4. Batch Generation
    print("\n--- Batch Generation ---")
    batch_input = torch.randint(0, 50257, (4, 5)).to(device)
    batch_output = model.generate(
        batch_input,
        max_length=15,
        do_sample=True
    )
    print(f"Batch output shape: {batch_output.shape}")

if __name__ == "__main__":
    main()
