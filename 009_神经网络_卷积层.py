import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

'''
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', device=None, dtype=None)
in_channels     输入图像通道数(int) 
out_channels    输出图像通道数(int) 
kernel_size     卷积核大小(int or tuple) 
stride          滑动窗口步长(int or tuple, optional) 
padding         边缘填充(int, tuple or str, optional)
padding_mode    边缘填充模式(str, optional)
dilation        内核元素之间的间距(int or tuple, optional) 
groups          从输入通道到输出通道的阻塞连接数(int, optional)
bias            如果为True，则向输出添加可学习的偏差(bool, optional) 
'''

dataset = torchvision.datasets.CIFAR10('related_data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

mynetwork = MyNetwork()
print(mynetwork)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = mynetwork(imgs)
    # print(output.shape) # [64, 6, 30, 30] batch_size channel H W

    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30)) # 因为输出图像为六通道无法显示，因此拆分成两个三通道图像，batch_size翻倍
    writer.add_images("ouput", output, step)

    step += 1

writer.close()