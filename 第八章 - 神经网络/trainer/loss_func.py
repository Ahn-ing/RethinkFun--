import torch

__package__ = "trainer"

# CrossEntropyLoss


class CrossEntropyLoss:
    def __init__(self, input: torch.Tensor, labels: torch.Tensor):
        self.N = labels.shape[0]
        self.y_hat = input
        # 给labels独热编码
        self.target = self.one_hot_encoding(labels)
        self.loss = self.compute_loss()

    # one_hot_encoding
    def one_hot_encoding(self, labels: torch.Tensor):
        y_one_hot = torch.zeros_like(self.y_hat)
        y_one_hot[torch.arange(self.N), labels] = 1  # 这里两个的形状一定要都是(n，)
        return y_one_hot

    def compute_loss(self):
        return -(self.target * torch.log(self.y_hat + 1e-8)).sum() / self.N
