import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(str(Path(__file__).resolve().parents[1]))

import sacrebleu

from .use_and_assess import TranslationModel


def get_bleu_score(model:TranslationModel, srcfile:str|Path, tgtfile:str|Path, device=None):
    if device is None:
        device = model.device

    srcfile, tgtfile = Path(srcfile), Path(tgtfile)
    with srcfile.open("r", encoding="utf-8") as f:
        src_sentences = [line.strip() for line in f if line.strip()]
    with tgtfile.open("r", encoding="utf-8") as f:
        tgt_sentences = [line.strip() for line in f if line.strip()]
    assert len(src_sentences) == len(tgt_sentences), "源文件和目标文件行数不一致"

    # 逐句翻译
    pred_sentences = []
    for i, src in enumerate(src_sentences):
        print("-" * 50)
        print(f"Translating {i+1}/{len(src_sentences)}...")
        pred = model(src)
        print(f"源句: {src}")
        print(f"参考译文: {tgt_sentences[i]}")
        print(f"模型译文: {pred}")
        print("-" * 50)
        pred_sentences.append(pred)
    
    # 计算 BLEU 分数
    bleu = sacrebleu.corpus_bleu(pred_sentences, [tgt_sentences], tokenize="zh")
    return bleu.score

if __name__ == "__main__":
    srcfile = Path(__file__).resolve().parents[2] / "chapter14" / "data14" / "en2cn" / "valid_en.txt"
    tgtfile = Path(__file__).resolve().parents[2] / "chapter14" / "data14" / "en2cn" / "valid_zh.txt"
    translator = TranslationModel()
    bleu_score = get_bleu_score(translator, srcfile, tgtfile, translator.device)
    print("\n" + "="*25 + " BLEU 评测结果 " + "="*25)
    print(f"BLEU score: {bleu_score:.2f}") # 百分制输出