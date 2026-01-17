import numpy as np  # noqa: F401
import torch

# # 1D Tensor
# t1 = torch.tensor([1, 2, 3], dtype=torch.float32)
# print(t1)

# # 2D Tensor
# t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(t2)

# # 3D Tensor
# t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(t3)

# # 从 NumPy 创建 Tensor
# arr = np.array([1, 2, 3])
# t_np = torch.tensor(arr)
# print(t_np)

test_tensor = torch.randint(0, 10, (3, 3))
arg_idx = test_tensor.argmax(dim=1)
max_idx = test_tensor.max(dim=1).indices
print(test_tensor)
print(arg_idx)
print(max_idx)
