import torch
from torch.utils.data import DataLoader, Dataset  # noqa: F401

__package__ = 'dataset' 

class MNISTDataset(Dataset):
    def __init__(self, filepath, mean=None, std=None):
        super().__init__()
        # 标准化
        images, labels = self._load_data(filepath)
        x = torch.tensor(images, dtype=torch.float32)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std(
                unbiased=False
            ).clamp_min(
                1e-8
            )  # 每个batch只关注当前分布，不需要去估计总体分布，不用无偏,后面则是设定最小值防止除以0
        self.mean = mean
        self.std = std
        x = (x - self.mean) / self.std

        self.x = x
        self.y = torch.tensor(labels, dtype=torch.int64)

    def _load_data(self, filepath):
        images = []
        labels = []

        with open(filepath, "r") as f:  # noqa: UP015
            next(f)  # 跳过第一行

            for line in f:
                sample = line.strip().split(",")
                images.append(
                    list(map(float, sample[1:]))
                )  # 输入是字符串需要处理，不然不能转化为tensor
                labels.append(list(map(int, sample[0])))
        return images, labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


# 定义数据加载器
batch_size = 64
trainfile = r"第八章 - 神经网络\mnist\mnist_train.csv\mnist_train.csv"
train_ds = MNISTDataset(trainfile)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
testfile = r"第八章 - 神经网络\mnist\mnist_test.csv\mnist_test.csv"
test_ds = MNISTDataset(testfile)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    x, y = next(iter(test_dl))  # 取第一个批次的数据查看
    print(x, y)
