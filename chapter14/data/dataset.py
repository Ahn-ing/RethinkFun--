import torch
import torch.nn as nn
from dictionary.build_dict import BOS_ID, EOS_ID, PAD_ID, tokenize_cn, tokenize_en
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, max_len=100):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.pairs = self.data_process(src_file, trg_file, max_len)

    def data_process(self, src_file, trg_file, max_len=100):
        pairs = []
        with open(src_file, encoding="utf-8") as f:
            src_lines = f.read().splitlines()
        with open(trg_file, encoding="utf-8") as f:
            trg_lines = f.read().splitlines()
        if len(src_lines) != len(trg_lines):
            raise ValueError(
                f"src/trg 行数不一致: src={len(src_lines)} trg={len(trg_lines)}"
            )

        for src, trg in zip(src_lines, trg_lines):
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]
            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                pairs.append((src_ids, trg_ids))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            trg_ids, dtype=torch.long
        )

    # 对一个batch的输入和输出token序列，依照最长的序列长度，
    # 用<pad> token进行填充，确保一个batch的数据形状一致，组成一个tensor。
    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens


if __name__ == "__main__":
    train_en_file = r"chapter14\data\en2cn\train_en.txt"
    train_zh_file = r"chapter14\data\en2cn\train_zh.txt"
    dataset = TranslationDataset(train_en_file, train_zh_file, tokenize_en, tokenize_cn)
    demo_dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn
    )
    for src, trg, _, _ in demo_dataloader:
        print(src.shape, trg.shape)
        print(src, trg)
        break
