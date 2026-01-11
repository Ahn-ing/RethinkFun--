import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from data import ImageDataset
from model import CNNModel
from torch.utils.data import DataLoader
from torchvision import transforms

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNModel().to(device)
BATCH_SIZE = 64
IMG_SIZE = 128
data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
epochs = 10
lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr)
data_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ]
)

# 加载数据，划分数据集
data_dir = r"第十章 - 卷积神经网络\data\PetImages"
dataset = ImageDataset(data_dir, data_transform)
train_ds, test_ds = dataset.splitData()
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

# 开始训练
for epoch in range(epochs):
    model.train() # 放在每个epoch前保险，以防在每个epoch后进行验证

    correct = 0
    total = 0
    acc_loss = 0.0
    for step, (x, label) in enumerate(train_dl):
        optimizer.zero_grad()
        x = x.to(device)
        label = label.view(-1).to(device,dtype=torch.float32)

        y_pred = model(x).view(-1)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        pred = (y_pred>0.5).float()
        correct += (pred == label).sum().item()
        total += label.numel()

        if (step+1)%100 == 0:
            avg_loss = acc_loss/100
            print(f'Step: {step+1} Loss: {avg_loss:.4f}')
            acc_loss = 0.0
    acc = (correct/total)*100
    print()
    print(f'epoch: {epoch+1} Accuracy: {acc:2f}%')
    print()


