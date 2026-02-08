import torch
import torch.nn as nn

__package__ = "components"

from .FeedForwardBlock import FFN
from .LayerNorm import LN
from .multi_head_attention import MHA


class ResidualConnection(nn.Module):
    def __init__(self, hid_dim:int, droupout:float):
        super().__init__()
        self.layer_norm = LN(hid_dim)  # 层归一化，保持输入输出维度一致
        self.dropout = nn.Dropout(droupout) # 防止过拟合

    def forward(self, x:torch.Tensor, sublayer:MHA|FFN):
        # x [B, S, H], sublayer_output [B, S, H]
        sublayer_output = sublayer(self.layer_norm(x))  # 先归一化再通过子层
        return x + self.dropout(sublayer_output)  # 残差连接并应用 dropout