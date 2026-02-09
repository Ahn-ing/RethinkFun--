import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from components import APE, FFN, LN, MHA, ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(self,hid_dim:int, n_heads:int):
        super().__init__()
        self.self_atten = MHA(hid_dim, n_heads, dropout=0.1)
        self.cross_atten = MHA(hid_dim, n_heads, dropout=0.1)
        self.ffn = FFN(hid_dim, pf_dim=hid_dim*4, droupout=0.1)
        self.residual_self_atten = ResidualConnection(hid_dim, droupout=0.1)
        self.residual_cross_atten = ResidualConnection(hid_dim, droupout=0.1)
        self.residual_ffn = ResidualConnection(hid_dim, droupout=0.1)

    def forward(self, x:torch.Tensor, enc_output:torch.Tensor, tgt_mask, src_mask):
        # x [B, S_tgt, H], enc_output [B, S_src, H], src_mask [B, 1, 1, S_src], tgt_mask [B, 1, S_tgt, S_tgt]
        x = self.residual_self_atten(x, lambda x: self.self_atten(x, x, x, tgt_mask))  # [B, S_tgt, H]
        x = self.residual_cross_atten(x, lambda x: self.cross_atten(x, enc_output, enc_output, src_mask))# [B, S_tgt, H]
        x = self.residual_ffn(x, self.ffn)  # [B, S_tgt, H]
        return x  # [B, S_tgt, H]
    
class Decoder(nn.Module):
    def __init__(self, hid_dim:int, n_heads:int, max_len:int, n_layers:int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(hid_dim, n_heads) for _ in range(n_layers)
        ])
        self.ape = APE(max_len, hid_dim, dropout=0.1)
        self.layer_norm = LN(hid_dim)

    def forward(self, x:torch.Tensor, enc_output:torch.Tensor, tgt_mask, src_mask):
        x = self.ape(x)  # [B, S_tgt, H]
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)  # [B, S_tgt, H]
        x = self.layer_norm(x)
        return x  # [B, S_tgt, H]
