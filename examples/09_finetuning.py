"""
Example: Fine-tuning with BaseModel
===================================
Shows how to use BaseModel's fine-tuning capabilities.
"""

import torch
import torch.nn as nn
from composennent.basic import decoder, embedding, positional_encoding, SequentialBlock
from composennent.attention import causal_mask
from composennent.models import BaseModel


class SimpleLM(BaseModel):
    """Simple language model for fine-tuning demo."""
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.tok_emb = embedding("token", vocab_size=vocab_size, embed_dim=embed_dim)
        self.pos_enc = positional_encoding("absolute", d_model=embed_dim)
        self.layers = SequentialBlock(*[
            decoder("causal", latent_dim=embed_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)
        mask = causal_mask(x.size(1), device=x.device)
        x = self.layers(x, tgt_mask=mask)
        return self.head(self.norm(x))


def demo_freeze_finetune():
    """Freeze most layers and fine-tune only the head."""
    print("=== Freeze Fine-tuning ===")
    
    model = SimpleLM()
    print(f"Before freeze: {model.num_parameters(trainable_only=True):,} trainable")
    
    # Freeze all layers except the head
    model.freeze_layers()
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True
    
    print(f"After freeze: {model.num_parameters(trainable_only=True):,} trainable")
    print("Only the classification head is trainable!")


def demo_layer_specific_freeze():
    """Freeze specific layers by name."""
    print("\n=== Layer-Specific Freeze ===")
    
    model = SimpleLM()
    
    # Freeze only embeddings
    model.freeze_layers(layer_names=["tok_emb", "pos_enc"])
    
    trainable = model.num_parameters(trainable_only=True)
    total = model.num_parameters()
    print(f"Frozen embeddings: {trainable:,}/{total:,} trainable")


def demo_progressive_unfreeze():
    """Progressive unfreezing for transfer learning."""
    print("\n=== Progressive Unfreezing ===")
    
    model = SimpleLM()
    
    # Step 1: Freeze all
    model.freeze_layers()
    print(f"Step 1 - All frozen: {model.num_parameters(trainable_only=True):,} trainable")
    
    # Step 2: Unfreeze head
    for param in model.head.parameters():
        param.requires_grad = True
    print(f"Step 2 - Head unfrozen: {model.num_parameters(trainable_only=True):,} trainable")
    
    # Step 3: Unfreeze last layer
    for param in model.layers.layers[-1].parameters():
        param.requires_grad = True
    print(f"Step 3 - Last decoder unfrozen: {model.num_parameters(trainable_only=True):,} trainable")
    
    # Step 4: Unfreeze all
    model.unfreeze_layers()
    print(f"Step 4 - All unfrozen: {model.num_parameters(trainable_only=True):,} trainable")


def demo_quantization():
    """Quantize model for efficient inference."""
    print("\n=== Quantization ===")
    
    model = SimpleLM()
    model.eval()
    
    # Test inference before quantization
    x = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        out_fp32 = model(x)
    
    # Quantize to INT8
    quantized = model.quantize(bits=8, method="dynamic")
    
    # Test inference after quantization
    with torch.no_grad():
        out_int8 = quantized(x)
    
    print(f"FP32 output shape: {out_fp32.shape}")
    print(f"INT8 output shape: {out_int8.shape}")
    print("Model quantized successfully!")


def demo_save_load():
    """Save and load model checkpoints."""
    print("\n=== Save/Load Checkpoints ===")
    
    model = SimpleLM(vocab_size=500, embed_dim=64)
    
    # Save
    model.save("checkpoint.pt")
    print("Saved to checkpoint.pt")
    
    # Load
    loaded = SimpleLM.load("checkpoint.pt", vocab_size=500, embed_dim=64)
    print("Loaded from checkpoint.pt")
    
    # Verify
    x = torch.randint(0, 500, (1, 5))
    with torch.no_grad():
        out1 = model(x)
        out2 = loaded(x)
    
    match = torch.allclose(out1, out2)
    print(f"Outputs match: {match}")
    
    # Cleanup
    import os
    os.remove("checkpoint.pt")


if __name__ == "__main__":
    demo_freeze_finetune()
    demo_layer_specific_freeze()
    demo_progressive_unfreeze()
    demo_quantization()
    demo_save_load()
