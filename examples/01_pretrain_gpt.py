"""
Example: Pre-training a GPT model from scratch.

This script demonstrates:
1. Initializing a Transformer model configuration.
2. Setting up a tokenizer and dataset.
3. Using the model's .pretrain() method for pre-training.
"""

import torch
from torch.utils.data import Dataset

from composennent.nlp.transformers import GPT
from composennent.nlp.tokenizers import SentencePieceTokenizer

# --- Configuration ---
ModelConfig = {
    "vocab_size": 32000,
    "latent_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "drop_out": 0.1
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "lr": 6e-4,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- Mock Dataset ---
# In a real scenario, use composennent.utils.TextDataset or similar
class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer, length=1000):
        self.tokenizer = tokenizer
        self.length = length
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Simulating text data
        return "To be, or not to be, that is the question."

def main():
    print(f"Initializing model with config: {ModelConfig}")
    
    # 1. Initialize Tokenizer (Mocking a pretrained one for demo)
    # tokenizer = SentencePieceTokenizer.from_pretrained("tokenizer.model")
    tokenizer = SentencePieceTokenizer(vocab_size=ModelConfig['vocab_size']) 
    # Monkey-patching for the demo since we don't have a real model file
    tokenizer.encode = lambda x: [1] + [hash(w) % 1000 for w in x.split()] + [2]
    tokenizer.decode = lambda x: " ".join(str(i) for i in x)
    tokenizer.pad_id = 0

    # 2. Initialize Model
    model = GPT(**ModelConfig)
    print(f"Model parameters: {model.num_parameters():,}")

    # 3. Prepare Data
    # In reality: texts = ["doc1", "doc2", ...] or a Dataset object
    texts = [f"Sample text sequence {i}" for i in range(1000)]

    # 4. Pre-train using the model's .pretrain() method
    # The .pretrain() method automates the training loop, optimizer creation,
    # and device management.
    print("Starting pre-training...")
    trainer = model.pretrain(
        texts=texts,
        tokenizer=tokenizer,
        model_type="causal_lm",  # GPT uses causal language modeling
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        epochs=TRAIN_CONFIG["epochs"],
        device=TRAIN_CONFIG["device"]
    )

    # 5. Save
    model.save("models/gpt_pretrained.pt")
    print("Training complete. Model saved to models/gpt_pretrained.pt")

if __name__ == "__main__":
    main()
