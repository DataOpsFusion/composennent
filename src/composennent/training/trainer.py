from typing import Optional
import torch
from torch.cuda.amp import autocast, GradScaler
from composennent.training.dataloader import (
    create_dataloader,
    Batch,
)


def train(
    model,
    texts,
    tokenizer,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 512,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 3e-4,
    device: str = "cuda",
    padding_strategy: str = "max_length",
    pad_token_id: int = 0,
    use_amp: bool = True,  # Mixed precision training
):
    """Training loop for language models.

    Args:
        model: The model to train (e.g., GPT, BERT)
        texts: List of text strings to train on
        tokenizer: Tokenizer instance
        epochs: Number of training epochs
        batch_size: Batch size
        max_length: Maximum sequence length
        optimizer: Optional custom optimizer. If None, uses AdamW with lr
        lr: Learning rate (used only if optimizer is None)
        device: Device to train on ("cuda" or "cpu")
        padding_strategy: "max_length" (simple) or "longest" (efficient)
        pad_token_id: Padding token ID for dynamic padding
        use_amp: Use automatic mixed precision (FP16) for 2x speedup (default: True)

    Example:
        >>> # Use default optimizer with mixed precision
        >>> train(model, train_texts, tokenizer, epochs=5)
        >>>
        >>> # Use custom optimizer without mixed precision
        >>> custom_opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> train(model, train_texts, tokenizer, optimizer=custom_opt, use_amp=False)
    """
    dataloader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
        padding_strategy=padding_strategy,
        pad_token_id=pad_token_id,
    )

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initialize gradient scaler for mixed precision (only used if use_amp=True)
    scaler = GradScaler(enabled=use_amp)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = Batch(
                input_ids=batch.input_ids.to(device, non_blocking=True),
                attention_mask=batch.attention_mask.to(device, non_blocking=True),
                labels=batch.labels.to(device, non_blocking=True),
            )

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                logits, loss = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )

            # Scaled backward pass (handles FP16 gradients)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")