import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from data10 import ImageDataset, SubsetWithTransform
from eval_func import evaluate
from model10 import CNNModel
from torch.utils.data import DataLoader
from torchvision import transforms


# Windows + DataLoader(num_workers>0) 的典型报错：
# 子进程用 spawn 启动时，训练脚本必须用 if __name__ == "__main__":
# 保护入口；否则在导入/执行阶段就会再次启动子进程，触发该异常。
def main():
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNModel().to(device)
    lr = 0.001
    epochs = 10
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    IMG_SIZE = 128
    BATCH_SIZE = 64
    data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_ds_transfrom = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    )
    valid_ds_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    )

    # 数据集
    data_dir = r"第十章 - 卷积神经网络\data\PetImages"
    dataset = ImageDataset(data_dir)
    train_subset, val_subset = dataset.splitData()
    train_ds = SubsetWithTransform(train_subset, train_ds_transfrom)
    val_ds = SubsetWithTransform(val_subset, valid_ds_transform)
    train_dl = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 训练+验证
    for epoch in range(epochs):
        model.train()

        acc_loss = 0.0
        for step, (x, labels) in enumerate(train_dl):
            optimizer.zero_grad()

            x = x.to(device, non_blocking=True)
            labels = labels.to(device).float().view(-1)
            y_pred = model(x).view(-1)
            loss = criterion(y_pred, labels)

            loss.backward()
            optimizer.step()

            acc_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg_loss = acc_loss / 100
                print(f"Step {step + 1}: Average Loss {avg_loss:.4f}")
                acc_loss = 0.0

        acc = evaluate(model, val_dl, device)
        print()
        print(f"epoch {epoch + 1}: Validation Accuracy {acc * 100:.2f}%")
        print()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
