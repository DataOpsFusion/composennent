"""
Simple workflow example using model.pretrain() and model.instruct()

This shows the simplest way to train models:
1. model.pretrain() for pre-training
2. model.instruct() or model.fine_tune() for instruction tuning
"""

from composennent.nlp.transformers import GPT

# Mock tokenizer for demo
class MockTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.pad_token_id = 0
        self.vocab_size = 32000

    def encode(self, text, add_special_tokens=True):
        return [1] + [hash(w) % 1000 for w in text.split()] + [2]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def main():
    # 1. Create model
    model = GPT(
        vocab_size=32000,
        latent_dim=512,
        num_heads=8,
        num_layers=6,
    )

    tokenizer = MockTokenizer()

    # 2. Pre-train (like your train() function)
    texts = ["text 1", "text 2", "text 3"] * 100

    model.pretrain(
        texts=texts,
        tokenizer=tokenizer,
        epochs=1,
        batch_size=8,
        max_length=512,
    )

    model.save("models/pretrained.pt")

    # 3. Instruction tune (using model.instruct or model.fine_tune)
    model = GPT.load("models/pretrained.pt")

    instruction_data = [
        {
            "instruction": "What is AI?",
            "input": "",
            "output": "AI is artificial intelligence..."
        },
    ] * 50

    # Can use either:
    # model.instruct(...) or model.fine_tune(...)
    model.instruct(
        data=instruction_data,
        tokenizer=tokenizer,
        epochs=1,
        batch_size=4,
    )

    model.save("models/instructed.pt")

    print("Done!")


if __name__ == "__main__":
    main()
