import torch


def create_masks(src:torch.Tensor, trg:torch.Tensor, pad_id:int):
    # 1. 编码器掩码
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_src]
    # 2. 解码器掩码
    # 2.1 目标序列的 padding 掩码
    tgt_pad_mask = (trg != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_tgt]
    # 2.2 目标序列的未来信息掩码
    tgt_len = trg.shape[1]
    futer_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=trg.device)).bool()  # [S_tgt, S_tgt]
    tgt_mask = tgt_pad_mask & futer_mask  # [B, 1, S_tgt, S_tgt], 这里会自动广播 futer_mask
    return src_mask, tgt_mask

