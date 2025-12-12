"""
Example: NLP - Build a GPT-style Language Model
================================================
Shows how to build a GPT-style causal language model from scratch.
"""

import torch
import torch.nn as nn
from composennent.basic import (
    decoder,
    embedding,
    positional_encoding,
    SequentialBlock,
)
from composennent.attention import causal_mask
from composennent.models import BaseModel


class GPTModel(BaseModel):
    """Simple GPT-style language model using composennent blocks."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Token embedding + positional encoding
        self.tok_emb = embedding("token", vocab_size=vocab_size, embed_dim=embed_dim)
        self.pos_enc = positional_encoding("absolute", d_model=embed_dim, dropout=dropout)
        
        # Stack of decoder layers (causal self-attention)
        self.layers = SequentialBlock(*[
            decoder("causal", latent_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.tok_emb.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)
        
        # Causal mask
        mask = causal_mask(seq_len, device=x.device)
        
        # Transformer layers
        x = self.layers(x, tgt_mask=mask)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get last max_len tokens
            context = generated[:, -self.max_len:]
            
            # Forward pass
            logits = self(context)
            
            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def demo():
    print("=== Building a GPT Model ===")
    
    # Create model
    model = GPTModel(
        vocab_size=1000,  # Small vocab for demo
        embed_dim=256,
        num_heads=4,
        num_layers=4,
    )
    
    print(model.summary())
    
    # Forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    print(f"\nInput: {input_ids.shape}")
    print(f"Output logits: {logits.shape}")
    
    # Generate
    prompt = torch.randint(0, 1000, (1, 5))  # 5 token prompt
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"\nGeneration: {prompt.shape} -> {generated.shape}")
    
    # Save/load
    model.save("demo_gpt.pt")
    print("\nModel saved to demo_gpt.pt")


if __name__ == "__main__":
    demo()
