import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        attention,
        emb_dim,
        hid_dim,
        en_vocab_size,
        zh_vocab_size,
        pad_id,
        n_layers=3,
        device=None,
    ):
        super().__init__()
        self.encoder = encoder(en_vocab_size, emb_dim, hid_dim, pad_id, n_layers)
        self.attention = attention(hid_dim)
        self.decoder = decoder(
            2 * hid_dim, hid_dim, emb_dim, zh_vocab_size, pad_id, self.attention
        )
        self.zh_vocab_size = zh_vocab_size
        self.PAD_ID = pad_id
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_lens: list[int]):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        # 用于储存解码结果
        outputs = torch.zeros((max_len, batch_size, self.zh_vocab_size)).to(
            device=self.device
        )
        # 调用encoder
        encoder_final_h, cur_h_states, cur_cells = self.encoder(src, src_lens)
        # 调用decoder
        input_token = trg[0]  # trg [S, B], trg[0] [B] 第一个token就是BOS_ID
        mask = src == self.PAD_ID
        for t in range(1, max_len):
            pred, cur_h_states, cur_cells, _ = self.decoder(
                input_token, cur_h_states, cur_cells, encoder_final_h, mask
            )
            outputs[t] = pred  # [B, zh_vocab_size]
            input_token = trg[t]

        return outputs  # [S_t, B, zh_vocab_size]
