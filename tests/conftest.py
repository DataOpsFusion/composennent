"""Shared pytest fixtures for composennent tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Use CPU for testing to avoid GPU requirements."""
    return "cpu"


@pytest.fixture
def mock_tokenizer():
    """Simple mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.pad_token_id = 0
            self.vocab_size = 1000
            self.eos_id = 2
            self.bos_id = 1
        
        def encode(self, text, add_special_tokens=True):
            """Convert text to token ids (simple hash-based for testing)."""
            tokens = [self.bos_id] + [hash(w) % 900 + 10 for w in text.split()] + [self.eos_id]
            return tokens
        
        def decode(self, ids, skip_special_tokens=False):
            """Convert ids back to text (placeholder)."""
            if skip_special_tokens:
                ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
            return " ".join(str(i) for i in ids)
        
        def __call__(self, texts, **kwargs):
            """Batch encode for compatibility."""
            if isinstance(texts, str):
                texts = [texts]
            encoded = [self.encode(t) for t in texts]
            max_len = kwargs.get("max_length", max(len(e) for e in encoded))
            
            # Pad sequences
            input_ids = []
            attention_mask = []
            for enc in encoded:
                pad_len = max_len - len(enc)
                input_ids.append(enc + [self.pad_id] * pad_len)
                attention_mask.append([1] * len(enc) + [0] * pad_len)
            
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
    
    return MockTokenizer()


@pytest.fixture
def sample_texts():
    """Sample training texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Neural networks can learn complex patterns.",
    ]


@pytest.fixture
def small_model_config():
    """Configuration for a small model suitable for CPU testing."""
    return {
        "vocab_size": 1000,
        "latent_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 128,
    }
