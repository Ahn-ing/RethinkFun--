import torch
import torch.nn as nn


# 手搓多头注意力机制
class MHA(nn.Module):
    def __init__(self, hid_dim:int, n_heads:int, dropout:float):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        # 将输入的 hid_dim 维度映射到 Q、K、V 三个不同的子空间, bias=False 提升计算效率
        self.fc_q = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias=False)
        # 将多头的输出重新映射回 hid_dim 维度
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout) # 防止过拟合
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))  # 缩放因子

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor|None):
        # query, key, value: [B, S, hid_dim], 这些输入是经过嵌入层和位置编码后的表示
        # mask: [B, 1, 1, S] or [B, 1, S, S] 要与注意力权重的形状匹配， 用于遮挡填充位置或未来信息
        B = query.shape[0]
        S_q = query.shape[1]
        S_k = key.shape[1]
        # 1. 线性变换并分割成多个头
        Q = self.fc_q(query)  # [B, S_q, hid_dim]
        K = self.fc_k(key)    # [B, S_k, hid_dim]
        V = self.fc_v(value)  # [B, S_k, hid_dim] ，注意这里 value 的时间步长度与 key 相同， 因为注意力是基于 key 来计算的， value 提供实际信息，key 是 value 的索引
        # 分割成 n_heads 个头，并调整形状以便并行计算， 调整形状因为注意力计算需要在头维度上进行
        Q = Q.view(B, S_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, S_q, head_dim]
        K = K.view(B, S_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, S_k, head_dim]
        V = V.view(B, S_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, S_k, head_dim]
        # 2. 计算注意力得分
        # Q 和 K 的点积，得到注意力得分矩阵
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)  # [B, n_heads, S_q, S_k]
        # 应用掩码（如果有的话）
        if mask is not None:
            # mask 中为0的位置表示需要被遮挡的部分， 将这些位置的能量值设为一个非常小的数， 避免它们在 softmax 中产生影响
            energy = energy.masked_fill(mask == 0, float("-1e9")) # mask [B, 1, 1, S_k] or [B, 1, S_q, S_k]
        # 3. 计算注意力权重
        attention = torch.softmax(energy, dim=-1)  # [B, n_heads, S_q, S_k]
        attention = self.dropout(attention)  # 应用 dropout 防止过拟合
        # 4. 计算加权和
        x = torch.matmul(attention, V)  # [B, n_heads, S_q, head_dim]
        # 5. 连接多个头的输出
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, S_q, n_heads, head_dim]
        x = x.view(B, S_q, self.hid_dim)  # [B, S_q, hid_dim]
        # 6. 最终线性变换
        x = self.fc_o(x)  # [B, S_q, hid_dim]
        return x  # 返回输出
    
if __name__ == "__main__":
    print("[multi_head_attention.py] __main__ entered")
    # 测试 MultiHeadAttention 模块
    BATCH_SIZE = 2
    SEQ_LEN = 5
    HID_DIM = 8
    N_HEADS = 2
    DROPOUT = 0.1

    mha = MHA(hid_dim=HID_DIM, n_heads=N_HEADS, dropout=DROPOUT)
    query = torch.randn(BATCH_SIZE, SEQ_LEN, HID_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, HID_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, HID_DIM)
    mask = None  # 可以根据需要创建掩码

    output = mha(query, key, value, mask)
    print("Output shape:", output.shape)          # 应该是 [BATCH_SIZE, SEQ_LEN, HID_DIM]
    