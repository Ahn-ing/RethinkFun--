import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset import test_dl, train_dl
from loss_func import CrossEntropyLoss
from model import MLP


# ReLU 的导数：shape 与 input 完全一致
def Relu_grad(input: torch.Tensor) -> torch.Tensor:
    return (input > 0).to(dtype=input.dtype)


# 计算loss对最后一层logits的梯度
def delt_final(y_hat: torch.Tensor, target: torch.Tensor):
    return (y_hat - target) / len(y_hat)  # 平均损失对最后一层的导数


# 定义反向传播过程
def backward(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    weights: torch.Tensor,
    y_hat: torch.Tensor,
    target: torch.Tensor,
):
    w_grad = [None] * len(weights)
    b_grad = [None] * len(weights)
    delt = delt_final(y_hat, target)
    for i in range(len(weights) - 1, -1, -1):
        if i != len(weights) - 1:
            delt = delt @ weights[i + 1].t() * Relu_grad(outputs[i])
        delt_w = inputs[i].t() @ delt
        delt_b = delt.sum(dim=0)
        w_grad[i] = delt_w
        b_grad[i] = delt_b
    return w_grad, b_grad


# 模型训练
# 导入训练集
train_set = train_dl
# 定义模型
model = MLP()
device = "cuda" if torch.cuda.is_available() else "cpu"
# 开始训练
epochs = 10
lr = 0.1

for epoch in range(epochs):
    correct = 0
    total_loss = 0
    total = 0
    for x, label in train_set:
        x = x.to(
            device, non_blocking=True
        )  # non_blocking=True 只有在 pin_memory=True 且拷贝是 CPU→GPU 时才可能带来收益；没 CUDA 的话它基本没效果，但保留也没坏处。
        y_hat = model._forward(x)  # 计算模型输出
        label = label.view(-1)
        Loss = CrossEntropyLoss(y_hat, label)  # 计算损失
        # loss = Loss.compute_loss().item() # .item() 会强制把结果同步回 CPU，打断 GPU 流水线，速度会明显掉。
        loss = Loss.compute_loss()
        y_one_hot = Loss.target  # 得到原标签独热编码
        total_loss += loss
        w_grad, b_grad = backward(
            model.a, model.z, model.w, y_hat, y_one_hot
        )  # 这里要传入独热编码后的label
        with torch.no_grad():  # 更新参数
            for i in range(len(model.w)):
                model.w[i] -= lr * w_grad[i]
                model.b[i] -= lr * b_grad[i]

        pred_idx = y_hat.argmax(1)  # 取概率最大的那个
        correct += (pred_idx == label).sum().item()
        total += label.numel()  # 是 number of elements 的意思：返回这个 tensor 里一共有多少个元素（一个 Python int）。
    acc = correct / total
    print(
        f"epoch {epoch + 1}: loss {total_loss / len(train_set)}"
    )  # len(train_set)是batch的数量
    print(f"Train_Accuracy: {acc * 100:.2f}%")
    print()


# 模型评估
with torch.no_grad():
    correct = 0
    total = 0
    for x, label in test_dl:
        x = x.to(device)
        label = label.view(-1).to(device)
        y_hat = model._forward(x)
        pred_idx = y_hat.argmax(dim=1)
        correct += (pred_idx == label).sum().item()
        total += label.numel()

    acc = correct / total
    print(f"Test_Accuracy: {acc * 100:.2f}%")
