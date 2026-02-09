import torch
import torch.nn as nn


class APE(nn.Module):
    def __init__(self, max_len: int, hid_dim: int, dropout: float):
        super().__init__()
        assert (
            hid_dim % 2 == 0
        ), "hid_dim must be even for sinusoidal positional encoding"
        self.max_len = max_len # 位置编码的最大长度，通常设置为模型能够处理的最大序列长度
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hid_dim)  # [max_len, hid_dim] 存储位置编码的矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1] 每一行是一个位置索引
        div_term = torch.exp(
            torch.arange(0, hid_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hid_dim) # 同等变换防止指数过大导致数值不稳定
        )  # div_term [hid_dim/2] 用于计算不同维度的频率
        # position*divterm [max_len, hid_dim/2] 每一行是一个位置索引乘以不同频率的结果
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用余弦函数
        # 增加batch维度，方便后续直接加到输入上
        pe = pe.unsqueeze(0)  # [1, max_len, hid_dim]
        # 将位置编码注册为模型的缓冲区，这样它不会被更新，但会随模型一起保存和加载
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x[B, S, H] 输入的序列表示， 位置编码需要与输入的时间步长度和隐藏维度匹配
        seq_len = x.size(1)
        # 将位置编码添加到输入上， 注意这里是直接加法， 位置编码提供了位置信息， 输入提供了内容信息，两者结合后每个位置的表示既包含内容又包含位置信息
        x = x + self.pe[:, :seq_len, :]  # type: ignore # [B, S, H] 位置编码根据输入的时间步长度进行切片
        return self.dropout(x)  # 应用 dropout 防止过拟合


