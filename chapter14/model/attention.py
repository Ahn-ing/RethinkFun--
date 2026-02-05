import sys
from pathlib import Path

import torch
import torch.nn as nn

__package__ = "model"
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from data14 import TranslationDataset
from torch.utils.data import DataLoader

from .encoder import Encoder


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # 两个全连接层，第一层降维，第二层输出logit
        self.fc = nn.Linear(hid_dim*2+hid_dim, hid_dim)
        # 输出代表注意力的logit值
        self.logit = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_init_h:torch.Tensor, encoder_final_h:torch.Tensor, mask):
        # 将decoder_init_h分别与encoder_final_h合并
        # decoder_init_h: [1, B, H]    encoder_final_h: [S, B, 2H]
        decoder_init_h = decoder_init_h.permute(1, 0, 2) # [B, 1, H]
        encoder_final_h = encoder_final_h.permute(1, 0, 2) # [B, S, 2H]

        src_len = encoder_final_h.shape[1]
        decoder_init_h = decoder_init_h.repeat(1, src_len, 1) # [B, S, H]

        union = torch.cat([decoder_init_h, encoder_final_h], dim=2) # [B, S, 3H]
        energy = torch.tanh(self.fc(union)) # [B, S, H]

        attention = self.logit(energy).squeeze(2)  # [B, S]
         # mask标志哪些位置为<pad>,对于填充的位置，注意力值为一个大的负值。这样经过softmax就为0。
        mask = mask.permute(1, 0) # [S, B] -> [B, S]
        attention[mask] = -1e10
        return torch.softmax(attention, dim=1) # [B, S]
    

if __name__ == "__main__":
    train_en_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_en.txt"
    train_zh_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_zh.txt"
    demo_dataset = TranslationDataset(train_en_file, train_zh_file)
    demo_dataloader = DataLoader(demo_dataset, batch_size=64, shuffle=True, collate_fn=demo_dataset.collate_fn)
    demo_src = None
    demo_lens = None
    for src, _, src_lens, _ in demo_dataloader:
        demo_src = src
        demo_lens = src_lens
        print(demo_src.shape)
        break
    demo_encoder = Encoder(16000, 16, 10, pad_id=1)
    demo_outputs, demo_hidden_concat, demo_cell_concat = demo_encoder(demo_src, demo_lens)
    # 初始化演示输入
    demo_decoder_init_h = torch.randn((1,64,10))
    demo_attention = Attention(10)
    demo_a = demo_attention(demo_decoder_init_h, demo_outputs, demo_src==1)
    print(demo_a.shape)
    print(demo_a)