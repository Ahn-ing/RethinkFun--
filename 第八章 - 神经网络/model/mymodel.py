import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义模型
class MLP:
    def __init__(self):  # 类型提示：torch.tensor 应为 torch.Tensor
        self.layers_dim = [28 * 28, 128, 128, 128, 64, 10]
        # 初始化参数
        self.w = []
        self.b = []
        self.a = []
        self.z = []
        self._init_param()

    # 初始化参数
    def _init_param(self):
        for i in range(len(self.layers_dim) - 1):
            fan_in, fan_out = (
                self.layers_dim[i],
                self.layers_dim[i + 1],
            )  # 合理利用继承关系
            w = torch.randn(fan_in, fan_out, device=device) * math.sqrt(2.0 / fan_in)# 注意公式

            self.w.append(w)
            b = torch.zeros(fan_out, device=device)
            self.b.append(b)

    # 定义线性层
    def linear_layer(self, inputs, w, b):
        return inputs @ w + b

    # 定义激活函数
    # ReLu
    def Relu(self, logits: torch.Tensor):
        return torch.clamp(logits, min=0)
        # 掩码实现
        # mask = logits<0
        # logits[mask] = 0
        # return logits

        # softmax

    def softmax(self, logits: torch.Tensor):
        l_exp = torch.exp(
            logits - logits.max(dim=1, keepdim=True).values  # keepdim=True方便广播
        )  # 定义域为非负数很安全，不会数值溢出
        return l_exp / l_exp.sum(dim=1, keepdim=True)
        # torch.max()会返回最大值和其所在索引

    def _forward(self, x):
        self.a = [x]
        self.z = []
        for i in range(len(self.w) - 1):
            x = self.linear_layer(x, self.w[i], self.b[i])
            self.z.append(x)
            x = self.Relu(x)
            self.a.append(x)
        x = self.linear_layer(x, self.w[-1], self.b[-1])
        self.z.append(x)
        x = self.softmax(x)
        self.a.append(x)
        return x


if __name__ == "__main__":
    demo_dl = dataset.train_dl
    x, _ = next(iter(demo_dl))
    model = MLP(x)
    y_demo = model.output
    print(y_demo)
