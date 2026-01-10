import os

from PIL import Image


# 处理图像文件
def verify_and_encode_images(image_folder):
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
                samples.append((file_path, classes_encoding[cls_dir]))
            except Exception:
                print(f'Warning: Skipping corrupted image {file_path}')

