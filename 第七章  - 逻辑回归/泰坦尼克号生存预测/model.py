import torch
import torch.nn as nn


# 定义逻辑回归模型
class LRM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
