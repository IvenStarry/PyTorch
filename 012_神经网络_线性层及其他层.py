import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

'''
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
in_features：输入特征（输入样本大小） (int)
out_features：输出特征（输出样本大小） (int)
bias：是否添加偏置项 (bool)
'''
dataset = torchvision.datasets.CIFAR10("related_data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)
    
    def forward(self, input):
        output = self.linear1(input)
        return output

mynetwork = MyNetwork()

for data in dataloader:
    imgs, targets = data
    # print(imgs.shape) # (64, 3, 32, 32)
    
    output = torch.flatten(imgs) # flatten 将输入转换成一维tensor
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape) # flatten(196608) reshape(1, 1, 1, 196608)
    
    output = mynetwork(output)
    # print(output.shape) # (1, 1, 1, 10)