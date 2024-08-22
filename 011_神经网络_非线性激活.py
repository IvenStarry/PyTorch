import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.relu1 = ReLU() # inplace：是否直接替换输入
    
    def forward(self, input):
        output = self.relu1(input)
        return output

input = torch.tensor([[1, -0.5], [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

mynetwork = MyNetwork()
output = mynetwork(input)
print(output)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.sigmoid1 = Sigmoid()
    def forward(self, input):
        output = self.sigmoid1(input)
        return output
mynetwork = MyNetwork()

dataset = torchvision.datasets.CIFAR10("related_data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = mynetwork(imgs)

    writer.add_images("input_sigmoid", imgs, step)
    writer.add_images("output_sigmoid", output, step)

    step += 1

writer.close()