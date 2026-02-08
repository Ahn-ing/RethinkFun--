from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_model(model:nn.Module, save_path:str|Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model:nn.Module, load_path:str|Path, device:torch.device):
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model state file not found at {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    
        
    
def save_checkpoint(model:nn.Module, optimizer:Optimizer, epoch:int, best_loss:float, save_path:str|Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": epoch, 
        "best_loss": best_loss, 
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(obj, save_path)

def load_checkpoint(model:nn.Module, optimizer:Optimizer, load_path:str|Path, device:torch.device):
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {load_path}")
    obj = torch.load(load_path, map_location=device)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    start_epoch = obj.get("epoch", -1) + 1
    best_loss = obj.get("best_loss", None)
    return start_epoch, best_loss