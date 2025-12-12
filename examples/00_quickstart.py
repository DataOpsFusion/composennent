"""
Quickstart: Train and Fine-tune Models with Simple Method Calls

This example shows the easiest way to train models using the new
unified interface where models have .train() and .fine_tune() methods.
"""

import torch
from composennent.nlp.transformers import GPT
from composennent.nlp.tokenizers import SentencePieceTokenizer


def quickstart_example():
    """Complete example: pre-train, save, load, fine-tune"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === STEP 1: Create a Model ===
    print("Creating GPT model...")
    model = GPT(
        vocab_size=32000,
        latent_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=512,
    )
    print(f"Model has {model.num_parameters():,} parameters")

    # === STEP 2: Setup Tokenizer ===
    # In production, use a properly trained tokenizer:
    # tokenizer = SentencePieceTokenizer.from_pretrained("tokenizer.model")

    # For this demo, we'll create a simple mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.pad_token_id = 0  # Added for fine-tuning compatibility
            self.vocab_size = 32000

        def encode(self, text, add_special_tokens=True):
            return [1] + [hash(w) % 1000 for w in text.split()] + [2]

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

    tokenizer = MockTokenizer()

    # === STEP 3: Pre-train the Model ===
    print("\n--- Pre-training ---")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ] * 100

    # Simple one-liner pre-training!
    model.pretrain(
        texts=texts,
        tokenizer=tokenizer,
        epochs=2,
        batch_size=16,
        lr=3e-4,
        device=device,
    )

    # === STEP 4: Save Model ===
    model.save("models/pretrained_gpt.pt")
    print("Model saved!")

    # === STEP 5: Load Model ===
    print("\n--- Loading model ---")
    loaded_model = GPT.load("models/pretrained_gpt.pt", device=device)

    # === STEP 6: Fine-tune on Instructions ===
    print("\n--- Fine-tuning ---")

    # Instruction data in Alpaca format
    instruction_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Translate to Spanish",
            "input": "Hello, how are you?",
            "output": "Hola, ¿cómo estás?"
        },
        {
            "instruction": "Summarize this text",
            "input": "Machine learning is a subset of AI that focuses on algorithms.",
            "output": "ML is AI-focused on algorithms."
        },
    ] * 50

    # Simple one-liner fine-tuning!
    loaded_model.fine_tune(
        data=instruction_data,
        tokenizer=tokenizer,
        epochs=2,
        batch_size=8,
        lr=5e-5,  # Lower learning rate for fine-tuning
        device=device,
        mask_prompt=True,  # Only compute loss on outputs
    )

    # === STEP 7: Save Fine-tuned Model ===
    loaded_model.save("models/finetuned_gpt.pt")
    print("\nFine-tuning complete!")
    print(f"Model saved to: models/finetuned_gpt.pt")

    # === STEP 8: Generate Text ===
    print("\n--- Generating text ---")
    prompt = tokenizer.encode("What is")
    generated = loaded_model.generate(
        input_ids=prompt,
        max_length=50,
        temperature=0.8,
        device=device
    )
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")


def different_model_types():
    """Examples with different model types (BERT, etc.)"""

    # For BERT-style masked language modeling
    from composennent.nlp.transformers import BERT

    bert_model = BERT(
        vocab_size=30522,
        latent_dim=768,
        num_heads=12,
        num_layers=12,
    )

    # BERT uses masked language modeling
    # bert_model.pretrain(
    #     texts=texts,
    #     tokenizer=bert_tokenizer,
    #     model_type="mlm",  # Masked language modeling
    #     epochs=3,
    # )

    print("BERT model created (training commented out)")


if __name__ == "__main__":
    print("=" * 60)
    print("Composennent Quickstart: Unified Training Interface")
    print("=" * 60)

    quickstart_example()

    print("\n" + "=" * 60)
    print("All done! Your models now have .pretrain() and .fine_tune()")
    print("=" * 60)
