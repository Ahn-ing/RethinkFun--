import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from model.Transformer import Transformer

from chapter14.data14 import (
    BOS_ID,
    EN_VOCAB_SIZE,
    EOS_ID,
    PAD_ID,
    ZH_VOCAB_SIZE,
    decode_cn,
    tokenize_en,
)

from .save_and_load import load_model

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID_DIM = 512
N_HEADS = 8
MAX_LEN = 128
N_LAYERS = 6
# 初始化模型
model_state_path = Path(__file__).parent.parent / "best_model" / "best_model.pth"
transformer_model = Transformer(
    hid_dim=HID_DIM,
    n_heads=N_HEADS,
    max_len=MAX_LEN,
    n_layers=N_LAYERS,
    en_vocab_size=EN_VOCAB_SIZE,
    zh_vocab_size=ZH_VOCAB_SIZE,
).to(device)
# 加载训练好的模型权重
load_model(transformer_model, str(model_state_path), device)


# 定义翻译模型类
class TranslationModel(nn.Module):
    def __init__(self, model: Transformer = transformer_model, device=device):
        super().__init__()
        self.model = model
        self.device = device
        self.model.eval()  # 切换到评估模式

    # 测试翻译, 贪心解码
    def forward(self, sentence: str, max_len: int = 100):
        # 处理输入,先编码，然后创建词向量矩阵，最终输入形状为 [1, S_src, H]
        enc_input_tokens = (
            [BOS_ID] + tokenize_en(sentence) + [EOS_ID]
        )  # [BOS, token1, token2, ..., EOS]
        enc_input = (
            torch.LongTensor(enc_input_tokens).unsqueeze(0).to(self.device)
        )  # [1, S_src]
        # 创建编码器掩码
        src_mask = (enc_input != PAD_ID).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, S_src]
        # 词嵌入
        enc_input = self.model.en_embedding(enc_input)  # [1, S_src, H]
        # 编码器前向传播
        with torch.no_grad():
            enc_output = self.model.encoder(enc_input, src_mask)  # [1, S_src, H]
            # 贪心解码
            dec_input_tokens = [BOS_ID]  # 初始解码输入 [BOS]
            for _ in range(max_len):
                # 处理解码器输入,形状为 [1, S_tgt, H]
                dec_input = (
                    torch.LongTensor(dec_input_tokens).unsqueeze(0).to(self.device)
                )  # [1, S_tgt]
                # 创建解码器掩码
                tgt_len = dec_input.shape[1]
                tgt_mask = (dec_input != PAD_ID).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, S_tgt]
                np_mask = torch.tril(
                    torch.ones((tgt_len, tgt_len), device=self.device)
                ).bool()  # [S_tgt, S_tgt]
                tgt_mask = tgt_mask & np_mask  # [1, 1, S_tgt, S_tgt]
                # 词嵌入
                dec_input = self.model.zh_embedding(dec_input)  # [1, S_tgt, H]
                # 解码器前向传播
                dec_output = self.model.decoder(
                    dec_input, enc_output, tgt_mask, src_mask
                )  # [1, S_tgt, H]
                output = self.model.fc_out(dec_output)  # [1, S_tgt, ZH_VOCAB_SIZE]
                # EOS的概率
                print(f"P_EOS: {output[0, -1, EOS_ID].item()}")
                # 取最后一个时间步的预测结果
                # [1, S_tgt, ZH_VOCAB_SIZE] -> [1, S_tgt]
                pre_token = output.argmax(-1)[:, -1].item()  # 标量
                dec_input_tokens.append(pre_token)
                # 遇到 EOS token 则停止解码
                if pre_token == EOS_ID:
                    break
        # 将输出的索引转换回中文句子
        output_sentence = decode_cn(dec_input_tokens[1:])  # 去掉BOS token
        return output_sentence


if __name__ == "__main__":
    translate_model = TranslationModel()
    while True:
        test_sentence = input("请输入英文句子进行翻译（输入exit退出）：")
        if test_sentence.lower() == "exit":
            break
        translation = translate_model(test_sentence, max_len=128)
        print(f"翻译结果: {translation}")
