"""
Example: Encoder-Decoder Architecture (T5-style)
=================================================
Shows how to build a sequence-to-sequence model with cross-attention.
"""

import torch
import torch.nn as nn
from composennent.basic import (
    encoder,
    decoder,
    embedding,
    positional_encoding,
    SequentialBlock,
    CrossAttentionDecoder,
)
from composennent.attention import causal_mask, padding_mask
from composennent.models import BaseModel


class Seq2SeqTransformer(BaseModel):
    """Encoder-Decoder Transformer for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Shared embeddings for encoder and decoder
        self.embedding = embedding("token", vocab_size=vocab_size, embed_dim=embed_dim)
        self.pos_enc = positional_encoding("absolute", d_model=embed_dim, dropout=dropout)
        
        # Encoder stack (bidirectional self-attention)
        self.encoder_layers = SequentialBlock(*[
            encoder("transformer", latent_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack (causal self-attention + cross-attention)
        self.decoder_layers = nn.ModuleList([
            decoder("cross_attention", latent_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence."""
        x = self.embedding(src_ids)
        x = self.pos_enc(x)
        x = self.encoder_layers(x, key_padding_mask=src_mask)
        return x
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode target sequence with cross-attention to encoder output."""
        x = self.embedding(tgt_ids)
        x = self.pos_enc(x)
        
        # Causal mask for autoregressive decoding
        seq_len = x.size(1)
        causal = causal_mask(seq_len, device=x.device)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x, _ = layer(
                x,
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=tgt_mask,
                memory_key_padding_mask=memory_mask,
            )
        
        return x
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src_ids: Source token IDs (batch, src_len)
            tgt_ids: Target token IDs (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target padding mask
        
        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        # Encode
        memory = self.encode(src_ids, src_mask)
        
        # Decode
        x = self.decode(tgt_ids, memory, tgt_mask, src_mask)
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def demo():
    print("=== Building a Seq2Seq Transformer ===")
    
    model = Seq2SeqTransformer(
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )
    
    print(model.summary())
    
    # Forward pass
    batch_size = 2
    src_ids = torch.randint(0, 1000, (batch_size, 20))  # Source: 20 tokens
    tgt_ids = torch.randint(0, 1000, (batch_size, 15))  # Target: 15 tokens
    
    logits = model(src_ids, tgt_ids)
    print(f"\nSource: {src_ids.shape}")
    print(f"Target: {tgt_ids.shape}")
    print(f"Output logits: {logits.shape}")


if __name__ == "__main__":
    demo()
