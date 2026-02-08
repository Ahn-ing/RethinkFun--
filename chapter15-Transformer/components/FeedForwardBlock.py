import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, hid_dim:int, pf_dim:int, droupout:float):
        super().__init__()
         # 第一个线性层，将输入的 hid_dim 维度映射到 pf_dim 维度, 
         # pf表示 position-wise feedforward
        self.fc1 = nn.Linear(hid_dim, pf_dim) 
        self.relu = nn.ReLU()  # 激活函数，增加模型的非线性表达能力
        # 第二个线性层，将 pf_dim 维度映射回 hid_dim 维度，保持输入输出维度一致，方便残差连接
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(droupout) # 防止过拟合
    
    def forward(self, x:torch.Tensor):
        # x [B, S, H]
        x = self.fc1(x)  # [B, S, pf_dim] 线性变换
        x = self.relu(x) # [B, S, pf_dim] 激活函数
        x = self.dropout(x) # [B, S, pf_dim] 应用 dropout 防止过拟合
        x = self.fc2(x)  # [B, S, hid_dim] 线性变换
        return x # [B, S, hid_dim] 
