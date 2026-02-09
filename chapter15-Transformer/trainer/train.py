import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
__package__ = "trainer"

import datetime
from argparse import Namespace

import torch
import torch.nn as nn
import wandb
from model.Transformer import Transformer
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from chapter14.data14 import EN_VOCAB_SIZE, PAD_ID, ZH_VOCAB_SIZE, TranslationDataset

from .create_masks import create_masks
from .save_and_load import load_checkpoint, save_checkpoint, save_model

# configure wandb
wandb.login(
    key="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"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Namespace(
    project_name="Transformer_Translation",
    # data parameters
    batch_size=256,
    # model parameters
    hid_dim=512,
    n_heads=8,
    max_len=128,
    n_layers=6,
    # training parameters
    n_epochs=10,
    learning_rate=1e-4,
    # checkpoint parameters
    checkpoint_path=str(
        Path(__file__).parent.parent / "checkpoints" / "last_checkpoint.pth"
    ),
    best_model_path=str(Path(__file__).parent.parent / "best_model" / "best_model.pth"),
)


# model components
def get_model_optimizer_criterion(config: Namespace):
    model = Transformer(
        hid_dim=config.hid_dim,
        n_heads=config.n_heads,
        max_len=config.max_len,
        n_layers=config.n_layers,
        en_vocab_size=EN_VOCAB_SIZE,
        zh_vocab_size=ZH_VOCAB_SIZE,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    return model, optimizer, criterion


# dataset and dataloader
def get_train_dataloader(config: Namespace):
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
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.transformer_collate_fn,
        num_workers=4,  # 4 个 worker 并行加载数据
        pin_memory=(device.type == "cuda"),  # 配合 non_blocking=True
        persistent_workers=True,  # 反复 epoch 时减少 worker 重建开销
        prefetch_factor=2,  # 每个 worker 预取 2 批数据
    )
    return train_dataloader


def train(
    model: Transformer,
    optimizer: Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    log_interval: int = 1000,
):
    model.train()
    model = model.to(device)

    epoch_loss = 0
    acc_loss = 0
    step_count = 0
    for step, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input = tgt[
            :, :-1
        ]  # [B, S_tgt-1] 去掉最后一个 token 作为解码器输入，让模型学会当以BOS作为输入时可以在适当的时候输出EOS
        tgt_output = tgt[:, 1:]  # [B, S_tgt-1] 去掉第一个 token 作为目标输出
        # 创建掩码
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_id=PAD_ID)

        optimizer.zero_grad()
        output = model(
            src, tgt_input, src_mask, tgt_mask
        )  # [B, S_tgt-1, ZH_VOCAB_SIZE]
        # 计算损失
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)  # [B*(S_tgt-1), ZH_VOCAB_SIZE]
        tgt_output = tgt_output.reshape(
            -1
        )  # [B*(S_tgt-1)], 去掉第一个 token BOS 作为目标输出
        loss = criterion(output, tgt_output)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 统计损失
        acc_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1
        # 每1000步打印一次当前平均损失
        if (step + 1) % log_interval == 0:
            avg_loss = acc_loss / step_count
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now} Step {step+1}, Avg Loss: {avg_loss:.4f}")
            wandb.log({ "train/loss_step": avg_loss})
            acc_loss = 0
            step_count = 0
    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    print("[train.py] __main__ entered")
    # dataset and dataloader
    train_dataloader = get_train_dataloader(config)
    # model components
    model, optimizer, criterion = get_model_optimizer_criterion(config)
    # ==========================================================================#
    # 初始化wandb
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project=config.project_name,
        config=vars(config),
        name=f"Transformer_Translation_{now_time}",
        save_code=True,
        id=f"run_{now_time}",
        resume="allow",
    )
    model.__dict__["run_id"] = run.id
    # ==========================================================================#
    # training loop
    # 检查是否存在之前的检查点，如果存在则加载
    # last_ckpt = load_checkpoint(ckpt_file)
    start_epoch = 0
    best_loss: float | None = float("inf")
    ckpt_file = Path(config.checkpoint_path)
    if ckpt_file.exists():
        print(f"Loading checkpoint from {ckpt_file}")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, ckpt_file, device=device
        )
    # 模型状态保存路径
    save_file = Path(config.best_model_path)

    for epoch in range(start_epoch, config.n_epochs):
        epoch_loss = train(model, optimizer, criterion, train_dataloader, device)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} Epoch {epoch+1}/{config.n_epochs}, Loss: {epoch_loss:.4f}")
        # 记录到 wandb
        wandb.log({ "train/loss_epoch": epoch_loss})
        # 每个 epoch 结束后保存检查点
        save_checkpoint(model, optimizer, epoch, epoch_loss, ckpt_file)
        # 如果当前 epoch 的损失比之前的最佳损失更好，则保存模型状态
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, save_file)
            print(f" New best model saved with loss {best_loss:.4f}")
    # 结束 wandb 运行
    run.finish()
