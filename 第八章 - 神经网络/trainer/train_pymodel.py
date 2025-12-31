import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from model import PyMLP
from dataset import train_dl, test_dl

# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs_num = 10
lr = 0.1

model  = PyMLP().to(device=device)
optimizer = torch.optim.SGD(model.parameters(),lr)
criterion = nn.CrossEntropyLoss()

# 模型训练

model.train()
for epoch in range(epochs_num):
    total_loss = 0
    correct = 0
    total = 0
    for x, labels in train_dl:
        optimizer.zero_grad()
        x, labels = x.to(device), labels.to(device).view(-1)

        y_pred = model(x)
        loss = criterion(y_pred, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_idx = y_pred.argmax(dim=1)
        correct += (pred_idx==labels).sum().item()
        total += labels.numel()
    
    avg_loss = total_loss/len(train_dl)
    acc = correct/total
    print(f'epoch {epoch+1}: loss {avg_loss:.4f}')
    print(f'Accuracy: {acc*100:.2f}%')
    print()

