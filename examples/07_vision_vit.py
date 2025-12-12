"""
Example: Vision - Build a Vision Transformer (ViT)
===================================================
Shows how to build a Vision Transformer for image classification.
"""

import torch
import torch.nn as nn
from composennent.basic import (
    encoder,
    embedding,
    positional_encoding,
    SequentialBlock,
)
from composennent.models import BaseModel


class VisionTransformer(BaseModel):
    """Vision Transformer (ViT) for image classification."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_emb = embedding(
            "patch",
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding (learnable for ViT)
        self.pos_enc = positional_encoding(
            "relative",  # Learnable positions
            d_model=embed_dim,
            max_len=self.num_patches + 1,  # +1 for CLS
            dropout=dropout,
        )
        
        # Transformer encoder layers
        self.layers = SequentialBlock(*[
            encoder("transformer", latent_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Image tensor of shape (batch, channels, height, width)
        
        Returns:
            Class logits of shape (batch, num_classes)
        """
        batch_size = images.size(0)
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_emb(images)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Transformer encoder
        x = self.layers(x)
        
        # Classification: use CLS token output
        cls_output = self.norm(x[:, 0])
        logits = self.head(cls_output)
        
        return logits


def demo():
    print("=== Building a Vision Transformer ===")
    
    # Create ViT-Tiny for demo
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=10,  # Small for demo
        embed_dim=192,
        num_heads=3,
        num_layers=4,
    )
    
    print(model.summary())
    
    # Forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    logits = model(images)
    print(f"\nInput images: {images.shape}")
    print(f"Output logits: {logits.shape}")
    
    # Predictions
    preds = logits.argmax(dim=-1)
    print(f"Predictions: {preds}")


if __name__ == "__main__":
    demo()
