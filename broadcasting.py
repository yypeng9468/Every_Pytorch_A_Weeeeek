# General Semantics
# 如果以下条件满足，则可以说两个 tensor 是 broadcastable 的
# 1.每个tensor至少是一维的
# 2.当对所有维度进行迭代的时候，从最后一个维度开始，tensor之间的维度大小要么相同，要么其中一个为1，要么其中一个为空

import torch

x = torch.empty(5, 7, 3)
y = torch.empty(5, 7, 3)
print((x+y).shape) # (5, 7, 3)

x = torch.empty((0,))
y = torch.empty(2,2)
try:
    print((x+y).shape)
except RuntimeError as identifier:
    print(identifier) # The size of tensor a (0) must match the size of tensor b (2) at non-singleton dimension 1

x = torch.empty(5, 3, 4, 1)
y = torch.empty(   3, 1, 1)
print((x+y).shape) # (5, 3, 4, 1)

x = torch.empty(5, 2, 4, 1)
y = torch.empty(   3, 1, 1)
try:
    print((x+y).shape)
except RuntimeError as identifier:
    print(identifier) # The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

# 如果两个 tensor 是 broadcastable 的，得到的结果的tensor的大小计算方法如下：
# 如果 x 和 y 的维度大小不一样，对维度较小的那个 tensor 添加1进行前置，使得他们的长度相等
# 然后对于每一个维度大小，则是那个维度上 x 和 y 的维度大小取最大值
x = torch.empty(5, 3, 4, 1)
y = torch.empty(   3, 1, 1)
(x + y).size() # (5, 3, 4, 1)

x = torch.empty(1)
y = torch.empty(3, 1, 7)
(x + y).size() # (3, 1, 7)

x = torch.empty(5, 2, 4, 1)
y = torch.empty(   3, 1, 1)
try:
    (x + y).size()
except RuntimeError as identifier:
    print(identifier)

x = torch.empty(5, 3, 4, 1)
y = torch.empty(   3, 1, 1)
(x.add_(y)).size() # (5, 3, 4, 1)

x = torch.empty(5, 2, 4, 1)
y = torch.empty(   3, 1, 1)
try:
    (x.add_(y)).size()
except RuntimeError as identifier:
    print(identifier)

# 老版本的 pytorch 允许对不同 shape 但是数目相同的 tensor 进行 pointwise 的函数运算
# 计算方法主要是将这些tensor看做一维的进行，自从支持了广播之后，当维度不同但是数目相同的tensor
# 进行运算时，这种 “1-dimensional” pointwise的行为会出现警告
(torch.add(torch.ones(4,1), torch.randn(4))).size()

torch.utils.backcompat.broadcast_warning.enabled=True
torch.add(torch.ones(4,1), torch.ones(4))





