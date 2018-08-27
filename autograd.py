import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

x = torch.randn(5,5)
y = torch.randn(5,5)
z = torch.randn((5,5), requires_grad=True)
a = x + y
print(a.requires_grad)
b = a + z 
print(b.requires_grad)

# 当你想要固定住你模型的某一部分或者你不会对某些东西进行求导的时候，这是很有用的。
# 比如你想对一个预训练好的CNN进行fine-tune
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# 替换最后一个fc层
model.fc = nn.Linear(512, 100)
# 只对分类器进行优化
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
