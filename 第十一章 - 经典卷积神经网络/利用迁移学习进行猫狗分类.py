import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "第十章 - 卷积神经网络"))
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
    
from data import ImageDataset, SubsetWithTransform  # noqa: E402
from trainer.eval_func import evaluate  # noqa: E402

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():  # 冻结所有层
    param.requires_grad = False
for param in model.layer4.parameters():  # 解冻第四阶段参数
    param.requires_grad = True
in_features = model.fc.in_features  # 更换分类头
model.fc = nn.Linear(in_features, 1)
model = model.to(device)

BATCH_SIZE = 64
IMG_SIZE = 128
DATA_MEAN, DATA_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EPOCHS = 10
LR = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), LR)

train_ds_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ]
)
valid_ds_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ]
)

# 导入数据集
data_dir = (
    r"C:\Users\12702\Desktop\RethinkFun深度学习\第十章 - 卷积神经网络\data\PetImages"
)
dataset = ImageDataset(data_dir)
train_subset, valid_subset = dataset.splitData()
train_ds = SubsetWithTransform(train_subset, train_ds_transform)
valid_ds = SubsetWithTransform(valid_subset, valid_ds_transform)
train_dataloader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_ds, shuffle=False)

# 训练+验证
for epoch in range(EPOCHS):
    model.train()

    acc_loss = 0.0
    for step, (x, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        x = x.to(device)
        labels = labels.to(device).float().view(-1)
        y_pred = model(x).view(-1)
        loss = criterion(y_pred, labels)

        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        if (step + 1) % 100 == 0:
            avg_loss = acc_loss / 100
            print(f"Step: {step + 1}, Average Loss: {avg_loss:.4f}")
            acc_loss = 0.0

    avg_acc = evaluate(model, valid_dataloader, device)
    print()
    print(f"epoch: {epoch + 1}, Average Accuracy: {avg_acc * 100:.2f}")
    print()
