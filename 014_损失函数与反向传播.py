import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Linear, Sequential, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

'''
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
reduction：指定要应用于输出的操作 none：不使用操作直接输出，mean：输出的总和将除以输出中的元素数量，sum：输出将被求和
input：(N, C) target：(N, C)
'''
loss = L1Loss()
result = loss(inputs, targets)
print(result)

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

'''
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
input：(N, C) target：(N, C)
'''
loss = MSELoss()
result = loss(inputs, targets)
print(result)

'''
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
input：(N, C)
'''
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

loss = CrossEntropyLoss()
result = loss(x, y)
print(result)

dataset = torchvision.datasets.CIFAR10("related_data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )
    
    def forward(self, x):
        x = self.model1(x)
        return x

loss = CrossEntropyLoss()
mynetwork = MyNetwork()

for data in dataloader:
    imgs, targets = data
    outputs = mynetwork(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward() # 反向传播