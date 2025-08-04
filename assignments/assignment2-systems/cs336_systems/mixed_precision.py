import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("fc1 output dtype:", x.dtype)
        x = self.ln(x)
        print("LayerNorm output dtype:", x.dtype)
        x = self.fc2(x)
        print("fc2 (logits) output dtype:", x.dtype)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToyModel(5, 5).to(device)
x = torch.randn(10, 5, dtype=torch.float32, device=device)
target = torch.randint(0, 5, (10,), device=device)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

with autocast(device_type=device, dtype=torch.bfloat16):
    y = model(x)
    print("Model output (logits) dtype:", y.dtype)
    loss = criterion(y, target)
    print("Loss dtype:", loss.dtype)



# 反向传播并打印梯度 dtype
scaler.scale(loss).backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Gradient of {name} dtype:", param.grad.dtype)


