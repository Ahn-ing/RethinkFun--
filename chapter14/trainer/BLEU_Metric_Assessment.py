import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(str(Path(__file__).parent.parent))

import sacrebleu

from .use_and_assess import TranslationModel


def assess_bleu(model:TranslationModel, src_file:str|Path, tgt_file:str|Path, device):
    src_file = Path(src_file)
    tgt_file = Path(tgt_file)
    with src_file.open("r", encoding="utf-8") as f:
        src_sentences = [line.strip() for line in f]
    with tgt_file.open("r", encoding="utf-8") as f:
        tgt_sentences = [line.strip() for line in f]
    assert len(src_sentences) == len(tgt_sentences), "源文件和目标文件行数不一致"

    pred_sentences = []
    for i, src in enumerate(src_sentences):
        print("-" * 50)
        print(f"Translating {i+1}/{len(src_sentences)}...")
        pred = model.translate_sentence(src)
        print(f"源句: {src}")
        print(f"参考译文: {tgt_sentences[i]}")
        print(f"模型译文: {pred}")
        print("-" * 50)
        pred_sentences.append(pred)
    
    bleu = sacrebleu.corpus_bleu(pred_sentences, [tgt_sentences], tokenize="zh")
    return bleu.score

if __name__ == "__main__":
    src_file = Path(__file__).parent.parent / "data14" / "en2cn" / "valid_en.txt"
    tgt_file = Path(__file__).parent.parent / "data14" / "en2cn" / "valid_zh.txt"
    translator = TranslationModel()
    bleu_score = assess_bleu(translator, src_file, tgt_file, translator.device)
    print("\n" + "="*25 + " BLEU 评测结果 " + "="*25)
    print(f"BLEU score: {bleu_score:.2f}") # 百分制输出



