import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TitanicDataset, splitFeatureCols
from model import LRM

# 读取数据
df = pd.read_csv(r'C:\Users\Anderson\Desktop\RethinkFun深度学习\第七章  - 逻辑回归\titanic\titanic_clean.csv')
feature_cols = splitFeatureCols(df)

# 定义数据集
dataset = TitanicDataset(df,feature_cols)

# 训练集与加载器
train_ds, valid_ds = dataset.splitData(0.1)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)


# 定义模型
input_dim = dataset.getInputDim()
model = LRM(input_dim)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# 训练模型
epochs = 100

model.train()
for epoch in range(epochs):
    correct = 0
    for x,labels in train_dl:
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失
        y_pred = model(x).squeeze(1)        # [batch,1] -> [batch]
        labels = labels.float().view(-1)    # 确保 [batch] 且是 float

        preds = (y_pred >= 0.5).float()
        correct += (preds == labels).sum().item()

        loss = nn.functional.binary_cross_entropy(y_pred, labels)
        # 反向传播
        loss.backward()
        optimizer.step()
    
    print(f'epoch {epoch+1}: loss {loss.item()}')
    print(f'Training Accuracy {correct/len(train_ds)}')
    print()

# 验证集评估模型性能
model.eval()
with torch.no_grad():
    valid_dl = DataLoader(valid_ds, batch_size=256, shuffle=True)
    correct = 0
    for x, labels in valid_dl:
        y_pred = model(x).squeeze(1)
        labels = labels.float().view(-1)

        preds = (y_pred >= 0.5).float()
        correct += (preds == labels).sum().item()

    print(f'Valid_Accuracy: {correct/len(valid_ds)}')



