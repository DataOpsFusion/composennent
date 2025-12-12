"""
Example: Building Blocks
========================
Shows how to use composennent's basic building blocks to create custom models.
"""

import torch
from composennent.basic import (
    # Factory functions
    encoder,
    decoder,
    embedding,
    positional_encoding,
    # Direct classes
    Block,
    SequentialBlock,
    TransformerEncoder,
    CausalDecoder,
)
from composennent.attention import causal_mask, padding_mask


def example_factory_functions():
    """Use factory functions to create components by type name."""
    print("=== Factory Functions ===")
    
    # Create encoder
    enc = encoder("transformer", latent_dim=256, num_heads=8)
    print(f"encoder('transformer') -> {type(enc).__name__}")
    
    # Create decoder (GPT-style causal or T5-style cross-attention)
    dec_causal = decoder("causal", latent_dim=256, num_heads=8)
    dec_cross = decoder("cross_attention", latent_dim=256, num_heads=8)
    print(f"decoder('causal') -> {type(dec_causal).__name__}")
    print(f"decoder('cross_attention') -> {type(dec_cross).__name__}")
    
    # Create embeddings
    tok_emb = embedding("token", vocab_size=10000, embed_dim=256)
    patch_emb = embedding("patch", image_size=224, patch_size=16, embed_dim=768)
    print(f"embedding('token') -> {type(tok_emb).__name__}")
    print(f"embedding('patch') -> {type(patch_emb).__name__}")
    
    # Create positional encodings
    pe_abs = positional_encoding("absolute", d_model=256)
    pe_rope = positional_encoding("rope", dim=64)
    print(f"positional_encoding('absolute') -> {type(pe_abs).__name__}")
    print(f"positional_encoding('rope') -> {type(pe_rope).__name__}")


def example_custom_block():
    """Create a custom Block component."""
    print("\n=== Custom Block ===")
    
    class ResidualMLP(Block):
        """Custom residual MLP block."""
        def __init__(self, dim: int, expansion: int = 4):
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim * expansion)
            self.fc2 = torch.nn.Linear(dim * expansion, dim)
            self.act = torch.nn.GELU()
            self.norm = torch.nn.LayerNorm(dim)
        
        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(self.norm(x))))
    
    block = ResidualMLP(256)
    x = torch.randn(2, 10, 256)
    out = block(x)
    print(f"ResidualMLP: {x.shape} -> {out.shape}")


def example_sequential_block():
    """Stack multiple blocks using SequentialBlock."""
    print("\n=== SequentialBlock ===")
    
    # SequentialBlock passes extra args (like masks) to Block layers
    model = SequentialBlock(
        encoder("transformer", latent_dim=256, num_heads=8),
        encoder("transformer", latent_dim=256, num_heads=8),
        encoder("transformer", latent_dim=256, num_heads=8),
    )
    
    x = torch.randn(2, 10, 256)
    out = model(x)
    print(f"3-layer encoder: {x.shape} -> {out.shape}")


def example_with_masks():
    """Use attention masks with encoders and decoders."""
    print("\n=== Attention Masks ===")
    
    # Create causal mask for autoregressive models
    seq_len = 10
    mask = causal_mask(seq_len)
    print(f"Causal mask shape: {mask.shape}")
    
    # Create padding mask
    lengths = torch.tensor([8, 10, 5])  # Actual lengths in batch
    pad_mask = padding_mask(lengths, max_len=10)
    print(f"Padding mask shape: {pad_mask.shape}")
    
    # Use with decoder
    dec = decoder("causal", latent_dim=256, num_heads=8)
    x = torch.randn(3, 10, 256)
    out = dec(x, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
    print(f"Decoder with masks: {x.shape} -> {out.shape}")


if __name__ == "__main__":
    example_factory_functions()
    example_custom_block()
    example_sequential_block()
    example_with_masks()
