import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from data14 import EN_VOCAB_SIZE, PAD_ID, ZH_VOCAB_SIZE, TranslationDataset
from model import Attention, Decoder, Encoder, Seq2Seq
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .save_and_load import load_checkpoint, save_checkpoint, save_model


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()
    model = model.to(device)
    epoch_loss = 0
    acc_loss = 0
    step_count = 0

    for step, batch in enumerate(dataloader):
        src, trg, src_lens, _ = batch  # src: [S, B], trg: [S_t, B]
         # non_blocking=True 配合 pin_memory=True 提升数据传输效率,
         # 主要作用是在使用 CUDA 时，允许异步数据传输，意思是数据传输和计算可以并行进行，减少等待时间。
        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(src, trg, src_lens)  # [S_t, B, zh_vocab_size]
        # 忽略第一个时间步的预测，因为它对应BOS token
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # [(S_t-1)*B, zh_vocab_size]
        trg = trg[1:].view(-1)  # [(S_t-1)*B]
        loss = criterion(output, trg)  #
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1

        # 每100步打印一次当前平均损失
        if (step + 1) % 100 == 0:
            print(f" Step {step+1}, Avg Loss: {acc_loss / step_count:.4f}")
            acc_loss = 0
            step_count = 0
    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    print("[train.py] __main__ entered")
    # cinfigurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512
    N_LAYERS = 3
    BATCH_SIZE = 32
    N_EPOCHS = 10
    LEARNING_RATE = 0.001
    # model components
    model = Seq2Seq(
        encoder=Encoder,
        decoder=Decoder,
        attention=Attention,
        emb_dim=ENC_EMB_DIM,
        hid_dim=HID_DIM,
        en_vocab_size=EN_VOCAB_SIZE,
        zh_vocab_size=ZH_VOCAB_SIZE,
        pad_id=PAD_ID,
        n_layers=N_LAYERS,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    # dataset and dataloader
    train_en_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_en.txt"
    train_zh_file = Path(__file__).parent.parent / "data14" / "en2cn" / "train_zh.txt"
    train_dataset = TranslationDataset(train_en_file, train_zh_file)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.rnn_collate_fn,
        num_workers=2,                      # 2 个 worker 并行加载数据
        pin_memory=(device.type == "cuda"), # 配合 non_blocking=True
        persistent_workers=True,            # 反复 epoch 时减少 worker 重建开销                
    )

    # save and load paths
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    model_state_dir = Path(__file__).parent.parent / "model_states"
    model_state_path = model_state_dir / "seq2seq_model.pt"
    # load last checkpoint if exists
    start_epoch = 0
    best_loss = float("inf")
    if last_ckpt_path.exists():
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, str(last_ckpt_path), device
        )
        print(f"Resuming training from epoch {start_epoch+1} with best loss {best_loss:.4f}")

    # training loop
    for epoch in range(start_epoch, N_EPOCHS):
        avg_loss = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Average Loss: {avg_loss:.4f}")

        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model=model, path=str(model_state_path))
            print(f" New best model saved with loss {best_loss:.4f}")
        # save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            path=str(last_ckpt_path),
        )

