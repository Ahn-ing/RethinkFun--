import sys
from pathlib import Path

__package__ = "model"
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from data14 import EN_VOCAB_SIZE, PAD_ID, ZH_VOCAB_SIZE, TranslationDataset
from torch.utils.data import DataLoader

from .attention import Attention
from .encoder import Encoder
from .seq2seq import Seq2Seq


class Decoder(nn.Module):
    def __init__(
        self, att_dim, hid_dim, emb_dim, zh_vocab_size, pad_id, attention, n_layers=3
    ):
        super().__init__()
        # 定义单向LSTM
        self.sin_lstm = nn.LSTM(emb_dim + att_dim, hid_dim, n_layers)
        # 定义分类头，需要每个时间步最后一层输出与该时间步对应的注意力向量合并
        self.fc = nn.Linear(hid_dim + att_dim, zh_vocab_size)
        # 中文词典词嵌入
        self.embedding = nn.Embedding(zh_vocab_size, emb_dim, padding_idx=pad_id)
        # 注意力类
        self.attention = attention  # 在外面初始化，这里传入的是已经初始化好的

    def forward(self, input_token, last_hid_states, last_cells, encoder_final_h, mask):
        # 处理第一个输入
        # input_token [B]
        input_token = input_token.unsqueeze(0)  # [1, B]
        decoder_x = self.embedding(input_token)  # [1, B, E]
        # 获取当前步对所有编码阶段时间步的注意力
        cur_init_h = last_hid_states[-1].unsqueeze(0)  # [1, B, H]
        a = self.attention(cur_init_h, encoder_final_h, mask)  # [B, S]
        a = a.unsqueeze(1)  # [B, 1, S]
        # 用矩阵乘法获取上下文注意力
        # encoder_final_h [S, B, 2H]
        encoder_final_h = encoder_final_h.permute(1, 0, 2)  # [B, S, 2H]
        att0 = torch.bmm(a, encoder_final_h).permute(
            1, 0, 2
        )  # [B, 1, 2H] -> [1, B, 2H]
        # 拼接为最终输入
        lstm_input = torch.cat([decoder_x, att0], dim=2)  # [1, B, E+2H]

        # 单一时间步最终隐藏状态
        cur_decoder_h, (cur_h_states, cur_cells) = self.sin_lstm(
            lstm_input, (last_hid_states, last_cells)
        )
        cur_decoder_h = cur_decoder_h.squeeze(0)
        # cur_decoder [1, B, H] -> [B, H]
        fc_input = torch.cat([cur_decoder_h, att0.squeeze(0)], dim=1)  # [B, 3H]
        pred = self.fc(fc_input)  # [B, zh_vocab_size]
        return (
            pred,
            cur_h_states,
            cur_cells,
            a.unsqueeze(1),
        )  # attention weights for visualization


if __name__ == "__main__":
    # config
    batch_size = 64
    emb_dim = 512
    hid_dim = 256
    # dataset
    train_en_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_en.txt"
    train_zh_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_zh.txt"
    demo_dataset = TranslationDataset(train_en_file, train_zh_file)
    demo_dataloader = DataLoader(
        demo_dataset, batch_size, shuffle=True, collate_fn=demo_dataset.collate_fn
    )
    demo_src:torch.Tensor
    demo_lens = None
    demo_trg:torch.Tensor
    for src, trg, src_lens, _ in demo_dataloader:
        demo_src = src
        demo_trg = trg
        demo_lens = src_lens
        break
    # model
    demo_model = Seq2Seq(
        Encoder,
        Decoder,
        Attention,
        emb_dim,
        hid_dim,
        EN_VOCAB_SIZE,
        ZH_VOCAB_SIZE,
        PAD_ID,
    )
    demo_output = demo_model(demo_src, demo_trg, demo_lens)
    print([demo_trg.shape[0], batch_size, ZH_VOCAB_SIZE])
    print(demo_output.shape)
