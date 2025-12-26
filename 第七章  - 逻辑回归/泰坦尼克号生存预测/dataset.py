import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

g = torch.Generator().manual_seed(26)
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
subset = ["Age", "Embarked"]
norm_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
enc_cols = ["Sex", "Embarked"]


# 数据清洗
def load_data(filepath=None):
    if filepath == None:
        candidates = [
            Path("titanic")
            / "train.csv",  # 推荐：相对路径（从“第七章  - 逻辑回归”目录启动时可用）
            Path(
                r"C:\Users\Anderson\Desktop\RethinkFun深度学习\第七章  - 逻辑回归\titanic\train.csv"
            ),  # 兜底：绝对路径
        ]
        for p in candidates:
            if p.exists():
                train_csv_path = p
                break
        else:
            raise FileNotFoundError("找不到 train.csv，请手动传入 train_csv_path")

    return pd.read_csv(train_csv_path)


def clean_data(df: pd.DataFrame, drop_cols, st):
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=st)
    return df


# 独热编码
def one_hot_encoding(df: pd.DataFrame, enc_cols):
    df = pd.get_dummies(df, columns=enc_cols, drop_first=True, dtype=int)
    return df


# 数据标准化
def normalizeData(df: pd.DataFrame, norm_cols):
    for col in norm_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


# 划分输入输出
def splitFeatureCols(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "Survived"]
    return feature_cols


# 定义Dataset类
class TitanicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, label_col="Survived"):
        super().__init__()
        self.x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col], dtype=torch.float32)

    # 获取数据集长度
    def __len__(self):
        return len(self.x)

    # 获取指定索引数据对
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    # 获取输入维度
    def getInputDim(self):
        return len(self.x[0])

    # 划分训练集，验证集
    def splitData(self, v_ratio):
        data_size = self.__len__()
        valid_ds_size = int(v_ratio * data_size)
        train_ds_size = data_size - valid_ds_size 
        # 需要整数
        return torch.utils.data.random_split(self,\
                                              [train_ds_size, valid_ds_size],\
                                                generator=g)


# 保存清洗好的数据
if __name__ == "__main__":
    df = load_data()  # 加载数据
    df = clean_data(df, drop_cols, subset)  # 清洗数据
    df = one_hot_encoding(df, enc_cols=enc_cols)  # 独热编码
    df = normalizeData(df, norm_cols=norm_cols)  # 数据标准化
    df.to_csv(
        Path(r"C:\Users\Anderson\Desktop\RethinkFun深度学习\第七章  - 逻辑回归\titanic")
        / "titanic_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )  # 保存处理好的数据
