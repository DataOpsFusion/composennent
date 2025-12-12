"""
Example: FFN Layers (SwiGLU, GEGLU, etc.)
=========================================
Shows the different FFN layer variants and how to use them.
"""

import torch
from composennent.ffn import SwiGLU, GEGLU, ReGLU, GLU, MLP


def demo_ffn_variants():
    """Compare different FFN layer variants."""
    print("=== FFN Layer Variants ===\n")
    
    in_features = 512
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)
    
    # Standard MLP (GELU activation)
    mlp = MLP(in_features=in_features)
    out = mlp(x)
    print(f"MLP: {x.shape} -> {out.shape}")
    print(f"  Used in: Classic Transformer, BERT")
    
    # SwiGLU (Swish + GLU)
    swiglu = SwiGLU(in_features=in_features)
    out = swiglu(x)
    print(f"\nSwiGLU: {x.shape} -> {out.shape}")
    print(f"  Used in: LLaMA, Mistral, Gemma")
    
    # GEGLU (GELU + GLU)
    geglu = GEGLU(in_features=in_features)
    out = geglu(x)
    print(f"\nGEGLU: {x.shape} -> {out.shape}")
    print(f"  Used in: T5, PaLM")
    
    # ReGLU (ReLU + GLU)
    reglu = ReGLU(in_features=in_features)
    out = reglu(x)
    print(f"\nReGLU: {x.shape} -> {out.shape}")
    print(f"  Used in: Some research models")
    
    # GLU (Sigmoid gate)
    glu = GLU(in_features=in_features)
    out = glu(x)
    print(f"\nGLU: {x.shape} -> {out.shape}")
    print(f"  Used in: Original GLU paper")


def demo_custom_dimensions():
    """Show how to customize hidden and output dimensions."""
    print("\n=== Custom Dimensions ===\n")
    
    # Default: hidden = 4x input, output = input
    swiglu_default = SwiGLU(in_features=512)
    
    # Custom: explicit hidden dimension
    swiglu_custom = SwiGLU(
        in_features=512,
        hidden_features=1024,  # 2x instead of 4x
        out_features=256,      # Downproject
    )
    
    x = torch.randn(2, 10, 512)
    
    out_default = swiglu_default(x)
    out_custom = swiglu_custom(x)
    
    print(f"Default SwiGLU(512): {x.shape} -> {out_default.shape}")
    print(f"Custom SwiGLU(512, hidden=1024, out=256): {x.shape} -> {out_custom.shape}")


def demo_parameter_count():
    """Compare parameter counts across variants."""
    print("\n=== Parameter Comparison ===\n")
    
    dim = 1024  # Typical hidden dim
    
    variants = {
        "MLP": MLP(dim),
        "SwiGLU": SwiGLU(dim),
        "GEGLU": GEGLU(dim),
        "ReGLU": ReGLU(dim),
        "GLU": GLU(dim),
    }
    
    for name, layer in variants.items():
        params = sum(p.numel() for p in layer.parameters())
        print(f"{name:8s}: {params:,} parameters")
    
    print("\nNote: GLU variants have ~33% more parameters due to the gate projection")


if __name__ == "__main__":
    demo_ffn_variants()
    demo_custom_dimensions()
    demo_parameter_count()
