import torch
from util import get_device
from typing import Callable

# Toy MLP
class MLP(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())
    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())
    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()
            # Backward
            y.backward()
    return run