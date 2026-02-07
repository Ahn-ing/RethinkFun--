import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hid_dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(hid_dim))
        self.beta = nn.Parameter(torch.zeros(hid_dim))

    def forward(self, x:torch.Tensor):
        x = x.float()  # [B, S, H] 确保输入是浮点数，避免整数除法问题
        mean = x.mean(dim=-1, keepdim=True)  # [B, S, 1] 计算最后一个维度的均值，保持维度以便后续广播
        std = x.std(dim=-1, keepdim=True)    # [B, S, 1] 计算最后一个维度的标准差，保持维度以便后续广播
        x_norm = (x - mean) / (std + self.eps)  # [B, S, H] 标准化输入，添加 eps 防止除以零
        return self.gamma * x_norm + self.beta  # [B, S, H]

