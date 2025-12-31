import torch.nn as nn


class PyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10), 
        )
    def forward(self, x):
        return self.model(x)


