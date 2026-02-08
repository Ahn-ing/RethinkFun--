import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
__package__ = "trainer"

import torch
import torch.nn as nn
from model.Transformer import Transformer
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from chapter14.data14 import EN_VOCAB_SIZE, PAD_ID, ZH_VOCAB_SIZE, TranslationDataset

from .create_masks import create_masks
from .save_and_load import load_checkpoint, save_checkpoint, save_model


def train(
    model: Transformer,
    optimizer: Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.train()
    model = model.to(device)

    epoch_loss = 0
    acc_loss = 0
    step_count = 0
    for step, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        src_mask, tgt_mask = create_masks(src, tgt, pad_id=PAD_ID)
        output = model(src, tgt, src_mask, tgt_mask)  # [B, S_tgt, ZH_VOCAB_SIZE]
        output_dim = output.shape[-1]
        # 忽略第一个时间步的预测，因为它对应BOS token
        output = (
            output[:, 1:, :].contiguous().view(-1, output_dim)
        )  # [B*(S_tgt-1), ZH_VOCAB_SIZE]
        tgt = tgt[:, 1:].contiguous().view(-1)  # [B*(S_tgt-1)]
        loss = criterion(output, tgt)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 统计损失
        acc_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1
        # 每1000步打印一次当前平均损失
        if (step + 1) % 1000 == 0:
            print(f" Step {step+1}, Avg Loss: {acc_loss / step_count:.4f}")
            acc_loss = 0
            step_count = 0
    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    print("[train.py] __main__ entered")
    # config
    BATCH_SIZE = 64
    HID_DIM = 512
    N_HEADS = 8
    MAX_LEN = 128
    N_LAYERS = 6
    N_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model components
    model = Transformer(
        hid_dim=HID_DIM,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        en_vocab_size=EN_VOCAB_SIZE,
        zh_vocab_size=ZH_VOCAB_SIZE,
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    # dataset and dataloader
    train_en_file = (
        Path(__file__).parent.parent.parent
        / "chapter14"
        / "data14"
        / "en2cn"
        / "train_en.txt"
    )
    train_zh_file = (
        Path(__file__).parent.parent.parent
        / "chapter14"
        / "data14"
        / "en2cn"
        / "train_zh.txt"
    )
    train_dataset = TranslationDataset(train_en_file, train_zh_file)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.transformer_collate_fn,
        num_workers=4,  # 4 个 worker 并行加载数据
        pin_memory=(DEVICE.type == "cuda"),  # 配合 non_blocking=True
        persistent_workers=True,  # 反复 epoch 时减少 worker 重建开销
    )
    # training loop
    # 检查是否存在之前的检查点，如果存在则加载
    # last_ckpt = load_checkpoint(ckpt_file)
    start_epoch = 0
    best_loss = float("inf")
    ckpt_file = Path(__file__).parent.parent / "checkpoints" / "last_checkpoint.pth"
    if ckpt_file.exists():
        print(f"Loading checkpoint from {ckpt_file}")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, ckpt_file, device=DEVICE
        )
    # 模型状态保存路径
    save_file = Path(__file__).parent.parent / "best_model" / "best_model.pth"

    for epoch in range(N_EPOCHS):
        epoch_loss = train(model, optimizer, criterion, train_dataloader, DEVICE)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss:.4f}")
        # 每个 epoch 结束后保存检查点
        save_checkpoint(model, optimizer, epoch, epoch_loss, ckpt_file)
        # 如果当前 epoch 的损失比之前的最佳损失更好，则保存模型状态
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, save_file)
            print(f" New best model saved with loss {best_loss:.4f}")
