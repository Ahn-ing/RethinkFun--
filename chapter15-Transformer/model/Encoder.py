import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from components import FFN, LN, MHA, RPE, ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, hid_dim:int, n_heads:int):
        super().__init__()
        self.atten = MHA(hid_dim, n_heads, dropout=0.1)
        self.ffn = FFN(hid_dim, pf_dim=hid_dim*4, droupout=0.1)
        self.residual_atten = ResidualConnection(hid_dim, droupout=0.1)
        self.residual_ffn = ResidualConnection(hid_dim, droupout=0.1)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x [B, S, H], mask [B, 1, S, S]
        x = self.residual_atten(x, lambda x: self.atten(x, x, x, mask)) # [B, S, H],这里使用lambda函数将self.atten包装成一个只接受x参数的函数，以符合ResidualConnection的接口要求
        x = self.residual_ffn(x, self.ffn) # [B, S, H]
        return x  # [B, S, H]
    
class Encoder(nn.Module):
    def __init__(self, hid_dim:int, n_heads:int, max_len:int, n_layers:int):
        super().__init__()
        self.rpe = RPE(max_len, hid_dim, dropout=0.1)
        self.layers = nn.ModuleList([
            EncoderBlock(hid_dim, n_heads) for _ in range(n_layers)
        ])
        self.layer_norm = LN(hid_dim)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x [B, S, H], mask [B, 1, S, S]
        x = self.rpe(x)  # [B, S, H]
        for layer in self.layers:
            x = layer(x, mask)  # [B, S, H]
        x = self.layer_norm(x)  # [B, S, H]
        return x  # [B, S, H]
    
if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 5
    HID_DIM = 8
    N_HEADS = 2
    MAX_LEN = 10
    N_LAYERS = 2

    encoder = Encoder(hid_dim=HID_DIM, n_heads=N_HEADS, max_len=MAX_LEN, n_layers=N_LAYERS)
    
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HID_DIM)
    mask = torch.ones(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)

    out = encoder(x, mask)
    print(out.shape)  # Expected output: torch.Size([2, 5, 8])
