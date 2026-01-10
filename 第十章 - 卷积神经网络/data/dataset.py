import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        super().__init__()
        self.samples = self.verify_and_encode_images(image_folder)
        self.transform = transform
        self.g = torch.Generator().manual_seed(42)
    # 处理图像文件
    def verify_and_encode_images(self, image_folder):
        classes = ['Cat', 'Dog'] # 类别名
        classes_encoding = {'Cat':0, 'Dog':1}  # 对应编码
        samples = []
        # 分别遍历各个文件夹的文件验证后编码
        for cls in classes:
            cls_dir = os.path.join(image_folder, cls) # 合并文件路径
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                # 尝试打开文件并验证
                file_path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # samples.append((file_path, classes_encoding[cls_dir])) 
                    # 这里应该传入的是cls，cls_dir是文件路径
                    samples.append((file_path, classes_encoding[cls])) 
                except Exception:
                    print(f'Warning: Skipping corrupted image {file_path}')
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label
    
    def splitData(self, train_ratio=0.8):
        data_size = self.__len__()
        train_size = int(data_size*train_ratio)
        test_size = data_size-train_size
        return random_split(self, [train_size, test_size], self.g)



if __name__ == "__main__":
    data_transform = transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dir = r'第十章 - 卷积神经网络\data\PetImages'
    demo_ds = ImageDataset(image_dir, data_transform)
    train_demo, test_demo = demo_ds.splitData()
    train_dl_demo = DataLoader(train_demo, 64, shuffle=True)
    x, y = next(iter(train_dl_demo))
    print(demo_ds.__len__())
    print(x, y)
