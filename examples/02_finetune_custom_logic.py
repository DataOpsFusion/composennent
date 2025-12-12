"""
Example: Instruction Fine-tuning (Two Approaches).

This script demonstrates:
1. Simple fine-tuning using model.fine_tune() method
2. Advanced fine-tuning with custom training logic by accessing the trainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from composennent.nlp.transformers import GPT
from composennent.instruct import InstructTrainer, InstructionDataset, InstructionCollatorWithPromptMasking
from composennent.nlp.tokenizers import BaseTokenizer

# --- Setup Mock Components ---

class SimpleTokenizer(BaseTokenizer):
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
    def encode(self, text): return [1] + [ord(c) % 1000 for c in text] + [2]
    def decode(self, ids, **kwargs): return "".join([chr(i) if i > 10 else '' for i in ids])

# Sample Instruction Data
dataset_samples = [
    {"instruction": "Summarize", "input": "Long text...", "output": "Short text."},
    {"instruction": "Translate", "input": "Hello", "output": "Hola"},
] * 50

def simple_finetune():
    """Approach 1: Simple fine-tuning using model.fine_tune()"""
    print("\n=== Simple Fine-tuning (Approach 1) ===")

    tokenizer = SimpleTokenizer()
    model = GPT(vocab_size=1000, latent_dim=256, num_heads=4, num_layers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # One-liner fine-tuning!
    trainer = model.fine_tune(
        data=dataset_samples,
        tokenizer=tokenizer,
        epochs=2,
        batch_size=8,
        lr=1e-5,
        device=device,
        use_amp=False,  # Set True for GPU with AMP
        mask_prompt=True,  # Masks prompt tokens in loss
    )

    model.save("models/finetuned_gpt.pt")
    print("Simple fine-tuning complete!")
    print("Model saved to: models/finetuned_gpt.pt")


def custom_finetune():
    """Approach 2: Advanced fine-tuning with custom training logic"""
    print("\n=== Custom Fine-tuning (Approach 2) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model & Data
    tokenizer = SimpleTokenizer()
    model = GPT(vocab_size=1000, latent_dim=256, num_heads=4, num_layers=4)

    dataset = InstructionDataset(dataset_samples, tokenizer)
    collator = InstructionCollatorWithPromptMasking(tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 2. Initialize Trainer Manually
    trainer = InstructTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        use_amp=False
    )

    # 3. Define Custom Training Step
    # This function replaces the default inner loop logic
    def custom_training_step(batch):
        # Move data
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)

        # Forward pass
        outputs = trainer.model(input_ids=input_ids)

        # Custom Loss Logic (e.g., label smoothing)
        logits = outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Optional: Custom logging
        if trainer.global_step % 10 == 0:
            print(f"[Custom] Step {trainer.global_step} Loss: {loss.item():.4f}")

        return loss

    # 4. Inject Custom Logic
    trainer.train_step = custom_training_step

    # 5. Run Training
    print("Starting instruction tuning with custom step...")
    trainer.train(dataloader, epochs=2)

    print("Custom fine-tuning complete!")


def main():
    # Choose which approach to run
    print("Fine-tuning Examples")

    # Simple approach - recommended for most users
    simple_finetune()

    # Advanced approach - for custom training logic
    # custom_finetune()


if __name__ == "__main__":
    main()
