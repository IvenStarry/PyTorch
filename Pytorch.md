# PyTorch

**Github**：https://github.com/IvenStarry  
**学习视频网站**：  
B站小土堆PyTorch https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=6fd71d34326a08965fdb07842b0124a7


## 卷积操作
**卷积运算**:矩阵作内积计算
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408220948293.png)
边界填充
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408220951588.png)

```python
import torch
import torch.nn.functional as F

'''
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
对由多个输入平面组成的输入图像应用2D卷积
input    输入图像
weight   权重(卷积核)
bias     偏置项
stride   滑动窗口步长 可以是一个数或者一个元组(sH,sW)
padding  边缘填充

'''
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1], 
                      [0, 1, 0],
                      [2, 1, 0]])

print(input.shape) # 不满足输入的尺寸要求
print(kernel.shape)

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape) # 满足输入的尺寸要求
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(output1)
output2 = F.conv2d(input, kernel, stride=2)
print(output2)
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
```

## 神经网络_卷积层
输入通道为1，输出通道为2，对原图像使用两个卷积核分别进行卷积操作，最终将卷积处理后的图像叠加在一起
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408220953318.png)

```python
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
```

## 神经网络_最大池化的使用
**最大池化**：保留神经网络的输入特征同时减小特征尺寸，从而减少计算量、降低模型复杂性并提高模型的鲁棒性
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408221533407.png)
**dilation参数**的意义：内核元素之间的间距
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408221532920.png)

```python
import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
                    return_indices=False, ceil_mode=False)
kernel_size     池化核大小(Union[int, Tuple[int, int]])
stride          滑动窗口步长，默认是池化核大小(Union[int, Tuple[int, int]])
padding         边缘填充(Union[int, Tuple[int, int]]
dilation        内核元素之间的间距(Union[int, Tuple[int, int]])
return_indices  如果为True，将返回最大索引和输出(bool)
ceil_mode       当为True时，将使用ceil而不是floor来计算输出形状(bool) (ceil向上取整 若窗口内输入元素不完整也保留 floor向下取整 若窗口内输入元素不完整则不保留)
'''

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5)) # 最大池化要求输入形状(N,C,H,W)

mynetwork = MyNetwork()
output = mynetwork(input)
print(output)


dataset = torchvision.datasets.CIFAR10("related_data", train=False, download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = mynetwork(imgs) # 池化操作不会改变channel，不需要reshape再输出
    
    writer.add_images("input_maxpool", imgs, step)
    writer.add_images("output_maxpool", output, step)
    
    step += 1

writer.close()
```

## 神经网络_非线性激活
**激活函数**：是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端，也是上层节点的输出和下层节点的输入之间具有一个函数关系
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408221611289.png)
```python

```
## 神经网络_线性层及其他层
|层名|作用|
|:-:|:-:|
|边缘填充层(Padding Layers)|设置边缘填充的方法类型|
|批标准化层(Normalization Layers)|加快神经网络的训练速度，防止过拟合|
|循环层(Recurrent Layers)|特定的网络结构，RNN、LSTM等框架，常用于文字识别中|
|变压器层(Transformer Layers)|特定网络中使用|
|线性层(Linear Layers)|对传入数据应用仿射线性变换 $y=xA^T+b$|
|失活层(Dropout Layers)|以概率p随机清零输入张量的一些元素,防止过拟合|
|Sparse Layers|特定网络中使用，自然语言处理|
|Distance Function|计算两个值的误差|
|Loss Function|误差计算|
|Flatten|将输入展开为一维张量|

线性层示意图
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408261546302.png)
```python
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
```

## 神经网络_搭建小实战和 Sequential 的使用
CIFAR-10网络结构与Padding、Stride参数计算
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408261549763.png)
Tensorboard可视化网络
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408261530858.png)
```python
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
'''
torch.nn.Sequential(*args: Module)
连接网络结构
'''

# 两种定义网络结构的方法
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


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


mynetwork = MyNetwork()
print(mynetwork)

input = torch.ones((64, 3, 32, 32)) # 全1张量
output = mynetwork(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(mynetwork, input) # 计算图
writer.close()
```

## 损失函数与反向传播
**梯度下降法**


```python

```

## 优化器
