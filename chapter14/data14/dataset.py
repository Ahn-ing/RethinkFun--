import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from array import array

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data14.dictionary.build_dict import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    tokenize_cn,
    tokenize_en,
)


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file,
        trg_file,
        src_tokenizer=tokenize_en,
        trg_tokenizer=tokenize_cn,
        max_len=100,
        cache_path: str | None = None,
    ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len

        cache_file = Path(cache_path) if cache_path else Path(__file__).parent / "cache" / "train_data.pt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if not cache_file.exists():
            packed = self.data_process(src_file, trg_file, max_len=100)
            self.src_tokens, self.src_offsets, self.trg_tokens, self.trg_offsets = packed
            torch.save(
                {
                    "src_tokens": self.src_tokens,
                    "src_offsets": self.src_offsets,
                    "trg_tokens": self.trg_tokens,
                    "trg_offsets": self.trg_offsets,
                    "max_len": max_len,
                },
                cache_file,
            )
        else:
            obj = torch.load(cache_file, map_location='cpu')
            self.src_tokens = obj["src_tokens"]
            self.src_offsets = obj["src_offsets"]
            self.trg_tokens = obj["trg_tokens"]
            self.trg_offsets = obj["trg_offsets"]
            
        

    def data_process(self, src_file, trg_file, max_len=100):
        with open(src_file, encoding="utf-8") as f:
            src_lines = f.read().splitlines()
        with open(trg_file, encoding="utf-8") as f:
            trg_lines = f.read().splitlines()
        if len(src_lines) != len(trg_lines):
            raise ValueError(
                f"src/trg 行数不一致: src={len(src_lines)} trg={len(trg_lines)}"
            )
        
        src_buf = array('I')
        trg_buf = array('I')
        src_offsets:list[int] = [0]
        trg_offsets:list[int] = [0]

        for src, trg in zip(src_lines, trg_lines):
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]
            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                src_buf.extend(src_ids)
                trg_buf.extend(trg_ids)
                src_offsets.append(len(src_buf))
                trg_offsets.append(len(trg_buf))
        
        src_tokens = torch.tensor(src_buf, dtype=torch.int32)
        trg_tokens = torch.tensor(trg_buf, dtype=torch.int32)
        src_offsets_t = torch.tensor(src_offsets, dtype=torch.int64)
        trg_offsets_t = torch.tensor(trg_offsets, dtype=torch.int64)
        return src_tokens, src_offsets_t, trg_tokens, trg_offsets_t

    def __len__(self):
        return self.src_offsets.numel()-1

    def __getitem__(self, idx):
        s0 = self.src_offsets[idx]
        s1 = self.src_offsets[idx+1]
        t0 = self.trg_offsets[idx]
        t1 = self.trg_offsets[idx+1] 
        src = self.src_tokens[s0:s1].to(dtype=torch.long)
        trg = self.trg_tokens[t0:t1].to(dtype=torch.long)
        return src, trg

    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_pad = nn.utils.rnn.pad_sequence(list(src_batch), padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(list(trg_batch), padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens


if __name__ == "__main__":
    train_en_file = r"chapter14\data14\en2cn\train_en.txt"
    train_zh_file = r"chapter14\data14\en2cn\train_zh.txt"
    dataset = TranslationDataset(
        train_en_file,
        train_zh_file,
        tokenize_en,
        tokenize_cn,
    )
    demo_dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn
    )
    for src, trg, _, _ in demo_dataloader:
        print(src.shape, trg.shape)
        print(src, trg)
        break
