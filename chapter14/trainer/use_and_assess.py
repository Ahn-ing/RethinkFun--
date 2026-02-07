import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(str(Path(__file__).parent.parent))

import torch
from data14 import (
    BOS_ID,
    EN_VOCAB_SIZE,
    EOS_ID,
    PAD_ID,
    ZH_VOCAB_SIZE,
    decode_cn,
    tokenize_en,
)
from model import Attention, Decoder, Encoder, Seq2Seq

from .save_and_load import load_model

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_DIM = 512
HID_DIM = 512
N_LAYERS = 3

# 初始化模型
model_state_path = Path(__file__).parent.parent / "model_states" / "seq2seq_model.pt"
seq2seq_model = Seq2Seq(
    Encoder,
    Decoder,
    Attention,
    EMB_DIM,
    HID_DIM,
    EN_VOCAB_SIZE,
    ZH_VOCAB_SIZE,
    PAD_ID,
    N_LAYERS,
    device,
).to(device)
# 加载训练好的模型权重
load_model(seq2seq_model, str(model_state_path), device)

# 定义翻译模型类
class TranslationModel:
    def __init__(self, model=seq2seq_model, device=device):
        self.model = model
        self.device = device
        self.model.eval()  # 切换到评估模式

    # 测试翻译, 贪心解码
    def translate_sentence(self, sentence, max_len=100):
        # 将输入句子转换为索引
        tokens = [BOS_ID] + tokenize_en(sentence) + [EOS_ID] # [BOS, token1, token2, ..., EOS]
        src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(self.device)  # [S, 1]
        src_len = [len(tokens)] # 注意这里是一个 batch，所以长度也是一个列表
        with torch.no_grad():
            # 先对输入进行编码
            enc_final_h, enc_h_states, enc_cells = self.model.encoder(src_tensor, src_len)
            # 初始化解码器的输入和隐藏状态
            input_token = torch.LongTensor([BOS_ID]).to(self.device)  # [1]
            last_h_states, last_cells = enc_h_states, enc_cells
            output_tokens = []
            for _ in range(max_len):
                mask = src_tensor == PAD_ID
                pred, last_h_states, last_cells, _ = self.model.decoder(input_token, last_h_states, last_cells, enc_final_h, mask)
                pred_token = pred.argmax(1).item()  # 取概率最高的token的索引
                if pred_token == EOS_ID:
                    break
                output_tokens.append(pred_token)
                input_token = torch.LongTensor([pred_token]).to(self.device)  # [1]
        
        # 将输出的索引转换回中文句子
        output_sentence = decode_cn(output_tokens)

        return output_sentence

if __name__ == "__main__":
    while True:
        test_sentence = input("请输入英文句子进行翻译（输入exit退出）：")
        if test_sentence.lower() == "exit":
            break
        if not test_sentence.strip(): # 如果输入为空，提示用户重新输入
            print("输入不能为空，请重新输入。")
            continue
        # 排除其他非英文字符的输入, 但可以包含正常标点符号和空格
        if any(char.isalpha() and not char.isascii() for char in test_sentence):# char.isalpha() 检查是否为字母， char.isascii() 检查是否为ASCII字符（即英文字符）
            print("输入包含非英文字符，请重新输入。")
            continue
        translator = TranslationModel()
        output_sentence = translator.translate_sentence(test_sentence)
        print(f"输入: {test_sentence}")
        print(f"输出: {output_sentence}")

