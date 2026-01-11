import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__package__ = "trainer"


def evaluate(model:nn.Module, dl:DataLoader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct = 0
    cnt = 0

    with torch.no_grad():
        for x, label in dl:
            x = x.to(device)
            label = label.to(device).view(-1).float()

            y_pred = model(x).view(-1)
            pred = (y_pred>0.5).float()
            correct += (pred == label).sum().item()
            cnt += label.numel()
        
        acc = correct/cnt
        return acc

