from composennent.basic import SequentialBlock
import torch
import torch.nn as nn


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
X = torch.rand(1, 28, 28, device=device)

model = SequentialBlock(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.Softmax(dim=1)
        )
model.to(device)
logits = model(X)


y_pred = logits.argmax(1)
print(f"Predicted class: {y_pred}")