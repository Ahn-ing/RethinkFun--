from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

__package__ = "trainer"

def save_model(model:nn.Module, path:str):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model:nn.Module, path:str, device:torch.device):
    load_path = Path(path)
    if load_path.exists():
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model state file not found at {path}")

def save_checkpoint(model:nn.Module, optimizer:Optimizer, epoch:int,  best_loss:float, path:str):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model:nn.Module, optimizer:Optimizer, path:str, device:torch.device):
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    chekpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(chekpoint["model_state_dict"])
    optimizer.load_state_dict(chekpoint["optimizer_state_dict"])
    start_epoch = chekpoint.get("epoch", -1) + 1
    best_loss = chekpoint.get("best_loss", None)
    return start_epoch, best_loss