from pathlib import Path

import sentencepiece as spm

here = Path(__file__).resolve().parent
corpus_dir = here.parent / 'en2cn'
en_file = corpus_dir / 'train_en.txt'
zh_file = corpus_dir / 'train_zh.txt'
en_model = here / 'en_bpe.model'
zh_model = here / 'zh_bpe.model'


sp_en = spm.SentencePieceProcessor()
sp_en.Load(str(en_model))
sp_cn = spm.SentencePieceProcessor()
sp_cn.Load(str(zh_model))

PAD_ID = sp_en.pad_id()
BOS_ID = sp_en.bos_id()
EOS_ID = sp_en.eos_id()

en_vocab = Path(__file__).parent.resolve() / "en_bpe.vocab"
zh_vocab = Path(__file__).parent.resolve() / "zh_bpe.vocab"

EN_VOCAB_SIZE = sp_en.GetPieceSize()
ZH_VOCAB_SIZE = sum(1 for line in zh_vocab.open("r", encoding="utf-8") if line.strip())


# -----------------------#
#        模型训练        #

#     spm.SentencePieceTrainer.Train(
#     f'--input={en_file} '
#     '--model_prefix=en_bpe '
#     '--vocab_size=16000 '
#     '--model_type=bpe '
#     '--character_coverage=1.0 '
#     '--unk_id=0 '
#     '--pad_id=1 '
#     '--bos_id=2 '
#     '--eos_id=3 '
# )

#     spm.SentencePieceTrainer.Train(
#         f'--input={zh_file} '
#         '--model_prefix=zh_bpe '
#         '--vocab_size=16000 '
#         '--model_type=bpe '
#         '--character_coverage=0.9995 '
#         '--unk_id=0 '
#         '--pad_id=1 '
#         '--bos_id=2 '
#         '--eos_id=3 '
#     )

def tokenize_en(text):
    return sp_en.Encode(text, out_type=int)

def tokenize_cn(text):
    return sp_cn.Encode(text, out_type=int)

if __name__ == "__main__":


    txt = '终于跑通字典构造了'

    encode_res = sp_cn.Encode(txt, out_type=int)
    print(f'编码： {encode_res}')

    decode_res = sp_cn.Decode(encode_res)
    print(f"解码：{decode_res}")
