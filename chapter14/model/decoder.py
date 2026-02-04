import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from attention import Attention
from data14 import BOS_ID, PAD_ID, TranslationDataset
from encoder import Encoder
from torch.utils.data import DataLoader


class Decoder(nn.Module):
    def __init__(self, att_dim, hid_dim, emb_dim, zh_vocab_size, pad_id, attention, n_layers=3):
        super().__init__()
        # 定义单向LSTM
        self.sin_lstm = nn.LSTM(emb_dim+att_dim, hid_dim, n_layers)
        # 定义分类头，需要每个时间步最后一层输出与该时间步对应的注意力向量合并
        self.fc = nn.Linear(hid_dim+att_dim, zh_vocab_size)
        # 中文词典词嵌入
        self.embedding = nn.Embedding(zh_vocab_size, emb_dim, padding_idx=pad_id)
        # 注意力类
        self.attention = attention(hid_dim)

    def forward(self, input_token, last_hid_states, last_cells, encoder_final_h, mask):
        # 处理第一个输入
        # input_token [B]
        input_token = input_token.unsqueeze(0) # [1, B]
        decoder_x = self.embedding(input_token) # [1, B, E]
        # 获取当前步对所有编码阶段时间步的注意力
        cur_init_h = last_hid_states[-1].unsqueeze(0) # [1, B, H]
        a = self.attention(cur_init_h, encoder_final_h, mask) # [B, S]
        a = a.unsqueeze(1) # [B, 1, S]
        # 用矩阵乘法获取上下文注意力
        # encoder_final_h [S, B, 2H]
        encoder_final_h = encoder_final_h.permute(1, 0, 2) # [B, S, 2H]
        att0 = torch.bmm(a, encoder_final_h).permute(1, 0, 2) # [B, 1, 2H] -> [1, B, 2H]
        # 拼接为最终输入
        lstm_input = torch.cat([decoder_x, att0], dim=2) # [1, B, E+2H]

        # 单一时间步最终隐藏状态
        cur_decoder_h, (cur_h_states, cur_cells) = self.sin_lstm(lstm_input, (last_hid_states, last_cells))
        cur_decoder_h = cur_decoder_h.squeeze(0)
        # cur_decoder [1, B, H] -> [B, H]
        fc_input = torch.cat([cur_decoder_h, att0.squeeze(0)], dim=1) # [B, 3H]
        pred = self.fc(fc_input)
        return pred, cur_h_states, cur_cells, a.squeeze(1) # attention weights for visualization


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
    demo_encoder = Encoder(16000, 16, 10)
    demo_outputs, demo_hidden_concat, demo_cell_concat = demo_encoder(demo_src, demo_lens)
    token_input = torch.tensor([BOS_ID]*64)
    demo_decoder = Decoder(20, 10, 16,16000, PAD_ID, Attention)
    packed = demo_decoder(token_input, demo_hidden_concat, demo_cell_concat, demo_outputs, demo_src==PAD_ID)
    demo_pred, _, _, _ = packed
    print(demo_pred.shape)
    print(demo_pred)





