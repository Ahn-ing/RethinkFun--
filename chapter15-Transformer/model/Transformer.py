import sys
from pathlib import Path

__package__ = "model"
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from .Decoder import Decoder
from .Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, hid_dim:int, n_heads:int, max_len:int, n_layers:int, en_vocab_size:int, zh_vocab_size:int):
        super().__init__()
        self.en_embedding = nn.Embedding(en_vocab_size, hid_dim)
        self.zh_embedding = nn.Embedding(zh_vocab_size, hid_dim)
        self.encoder = Encoder(hid_dim, n_heads, max_len, n_layers)
        self.decoder = Decoder(hid_dim, n_heads, max_len, n_layers)
        self.fc_out = nn.Linear(hid_dim, zh_vocab_size)
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src:torch.Tensor, trg:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor, cross_mask:torch.Tensor):
        # src [B, S_src], trg [B, S_tgt], src_mask [B, 1, S_src, S_src], cross_mask [B, 1, S_tgt, S_src]
        # 1. 词嵌入
        enc_input = self.en_embedding(src) # [B, S_src, H]
        dec_input = self.zh_embedding(trg) # [B, S_tgt, H]
        # 2. 编码器
        enc_output = self.encoder(enc_input, src_mask)  # [B, S_src, H]
        # 3. 解码器
        dec_output = self.decoder(dec_input, enc_output, tgt_mask, cross_mask)
        # 4. 输出层
        output = self.fc_out(dec_output)  # [B, S_tgt, zh_vocab_size]
        return output  # [B, S_tgt, zh_vocab_size]

if __name__ == "__main__":
    BATCH_SIZE = 2
    S_SRC = 5
    S_TGT = 6
    HID_DIM = 8
    N_HEADS = 2
    MAX_LEN = 10
    N_LAYERS = 2
    EN_VOCAB_SIZE = 100
    ZH_VOCAB_SIZE = 100

    model = Transformer(hid_dim=HID_DIM, n_heads=N_HEADS, max_len=MAX_LEN, n_layers=N_LAYERS, en_vocab_size=EN_VOCAB_SIZE, zh_vocab_size=ZH_VOCAB_SIZE)
    
    src = torch.randint(0, EN_VOCAB_SIZE, (BATCH_SIZE, S_SRC))  # [B, S_src]
    trg = torch.randint(0, ZH_VOCAB_SIZE, (BATCH_SIZE, S_TGT))  # [B, S_tgt]
    src_mask = torch.ones(BATCH_SIZE, 1, S_SRC, S_SRC)  # [B, 1, S_src, S_src]
    tgt_mask = torch.ones(BATCH_SIZE, 1, S_TGT, S_TGT)  # [B, 1, S_tgt, S_tgt]
    cross_mask = torch.ones(BATCH_SIZE, 1, S_TGT, S_SRC)  # [B, 1, S_tgt, S_src]

    output = model(src, trg, src_mask, tgt_mask, cross_mask)
    print(output.shape)  # Expected output: torch.Size([2, 6, zh_vocab_size])
