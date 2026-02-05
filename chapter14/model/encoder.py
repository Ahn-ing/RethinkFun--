import sys
from pathlib import Path

__package__ = "model"
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from data14 import PAD_ID, TranslationDataset
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_id, n_layers=3):
        super().__init__()
        self.PAD_ID = pad_id
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.bi_lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hid_dim, num_layers=self.n_layers, bidirectional=True
        )
        self.fc_hid = nn.ModuleList([(nn.Linear(hid_dim*2, hid_dim)) for _ in range(n_layers)])
        self.fc_cell = nn.ModuleList([(nn.Linear(hid_dim*2, hid_dim)) for _ in range(n_layers)])
    
    def forward(self, src:torch.Tensor, src_lens:list[int]):
        embeded = self.embedding(src)
        #将一个 padded sequence（已经填充到统一长度的 batch 序列） 转换为一个特殊的 PackedSequence 对象
        #这个对象在传入 RNN 时能跳过 padding 部分的计算。
        packed = nn.utils.rnn.pack_padded_sequence(embeded, src_lens, enforce_sorted=False)
        #(hidden, cell) ，形状都为(num_layers * 2, batch_size, hid_dim)表示每一层、每个方向在最后一个时间步的隐状态或细胞状态。
        outputs, (hidden, cell) = self.bi_lstm(packed)

        #将 PackedSequence 类型的输出还原成带 padding 的标准 Tensor，方便后续处理。
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, padding_value=self.PAD_ID)
        
        # 重塑隐藏状态和细胞状态: [n_layers * 2, batch, hid_dim] -> [n_layers, 2, batch, hid_dim]
        hidden:torch.Tensor = hidden.view(self.n_layers, 2, -1, hidden.size(2))
        cell:torch.Tensor = cell.view(self.n_layers, 2, -1, cell.size(2))

        # 合并每层最后一步的隐状态和细胞状态，并通过线性层降维
        final_hidden = []
        final_cell = []

        for layer in range(self.n_layers):
            hid_cat = torch.cat([hidden[layer][0], hidden[layer][1]], dim=1) # [B, H]
            cell_cat = torch.cat([cell[layer][0], cell[layer][1]], dim=1)
            hid_dim_red = self.fc_hid[layer](hid_cat).unsqueeze(0) # [1, B, H]
            cell_dim_red = self.fc_cell[layer](cell_cat).unsqueeze(0)

            final_hidden.append(hid_dim_red)
            final_cell.append(cell_dim_red)
        
        # 调整为可传递给Decoder的维度
        hidden_concat = torch.cat(final_hidden, dim=0) # [3, B, H]
        cell_concat = torch.cat(final_cell, dim=0)
        return outputs, hidden_concat, cell_concat

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
    demo_encoder = Encoder(16000, 16, 10, PAD_ID)
    demo_outputs, demo_hidden_concat, demo_cell_concat = demo_encoder(demo_src, demo_lens)
    print('outputs.shape', demo_outputs.shape)
    print('hidden.shape', demo_hidden_concat.shape)
    print('cell.shape', demo_cell_concat.shape)

