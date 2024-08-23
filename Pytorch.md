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
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
 
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)
 
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608,10)
    def forward(self,input):
        output = self.linear1(input)
        return output
 
tudui = Tudui()
 
for data in dataloader:
    imgs,targets = data
    print(imgs.shape)  #torch.Size([64, 3, 32, 32])
    # output = torch.reshape(imgs,(1,1,1,-1))  # 想把图片展平
    # print(output.shape)  # torch.Size([1, 1, 1, 196608])
    # output = tudui(output)
    # print(output.shape)  # torch.Size([1, 1, 1, 10])
    output = torch.flatten(imgs)   #摊平
    print(output.shape)   #torch.Size([196608])
    output = tudui(output)
    print(output.shape)   #torch.Size([10])
```

## 神经网络_搭建小实战和 Sequential 的使用
![alt text](image-3.png)
![alt text](image.png)
```python
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
 
 
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)  #第一个卷积
        self.maxpool1 = MaxPool2d(kernel_size=2)   #池化
        self.conv2 = Conv2d(32,32,5,padding=2)  #维持尺寸不变，所以padding仍为2
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()  #展平为64x4x4=1024个数据
        # 经过两个线性层：第一个线性层（1024为in_features，64为out_features)、第二个线性层（64为in_features，10为out_features)
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)  #10为10个类别，若预测的是概率，则取最大概率对应的类别，为该图片网络预测到的类别
    def forward(self,x):   #x为input
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
 
tudui = Tudui()
print(tudui)
```
```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
 
 
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
 
    def forward(self,x):   #x为input
        x = self.model1(x)
        return x
 
tudui = Tudui()
print(tudui)
 
input = torch.ones((64,3,32,32))  #全是1，batch_size=64,3通道，32x32
output = tudui(input)
print(output.shape)
```

## 损失函数与反向传播
**梯度下降法**
![alt text](image-1.png)
L1 loss
```python
import torch
from torch.nn import L1Loss
 
# 实际数据或网络默认情况下就是float类型，不写测试案例的话一般不需要加dtype
inputs = torch.tensor([1,2,3],dtype=torch.float32)   # 计算时要求数据类型为浮点数，不能是整型的long
targets = torch.tensor([1,2,5],dtype=torch.float32)
 
inputs = torch.reshape(inputs,(1,1,1,3))   # 1 batch_size, 1 channel, 1行3列
targets = torch.reshape(targets,(1,1,1,3))
 
loss = L1Loss()
result = loss(inputs,targets)
print(result)
```
均方误差
```python
import torch
from torch import nn
 
# 实际数据或网络默认情况下就是float类型，不写测试案例的话一般不需要加dtype
inputs = torch.tensor([1,2,3],dtype=torch.float32)   # 计算时要求数据类型为浮点数，不能是整型的long
targets = torch.tensor([1,2,5],dtype=torch.float32)
 
inputs = torch.reshape(inputs,(1,1,1,3))   # 1 batch_size, 1 channel, 1行3列
targets = torch.reshape(targets,(1,1,1,3))
 
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs,targets)
 
print(result_mse)
```
交叉熵
![alt text](image-2.png)
```python
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)
```

## 优化器
SGD优化器
```python
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
 
# 加载数据集并转为tensor数据类型
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
# 加载数据集
dataloader = DataLoader(dataset,batch_size=1)
 
# 创建网络名叫Tudui
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
 
    def forward(self,x):   # x为input，forward前向传播
        x = self.model1(x)
        return x
 
# 计算loss
loss = nn.CrossEntropyLoss()
 
# 搭建网络
tudui = Tudui()
 
# 设置优化器
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)  # SGD随机梯度下降法
 
for data in dataloader:
    imgs,targets = data  # imgs为输入，放入神经网络中
    outputs = tudui(imgs)  # outputs为输入通过神经网络得到的输出，targets为实际输出
    result_loss = loss(outputs,targets)
    optim.zero_grad()  # 把网络模型中每一个可以调节的参数对应梯度设置为0
    result_loss.backward()  # backward反向传播求出每一个节点的梯度，是对result_loss，而不是对loss
    optim.step()  # 对每个参数进行调优
```
```python
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
 
# 加载数据集并转为tensor数据类型
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)
 
# 创建网络名叫Tudui
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
 
    def forward(self,x):   # x为input，forward前向传播
        x = self.model1(x)
        return x
 
# 计算loss
loss = nn.CrossEntropyLoss()
 
# 搭建网络
tudui = Tudui()
 
# 设置优化器
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)  # SGD随机梯度下降法
for epoch in range(20):
    running_loss = 0.0  # 在每一轮开始前将loss设置为0
    for data in dataloader:  # 该循环相当于只对数据进行了一轮学习
        imgs,targets = data  # imgs为输入，放入神经网络中
        outputs = tudui(imgs)  # outputs为输入通过神经网络得到的输出，targets为实际输出
        result_loss = loss(outputs,targets)
        optim.zero_grad()  # 把网络模型中每一个可以调节的参数对应梯度设置为0
        result_loss.backward()  # backward反向传播求出每一个节点的梯度，是对result_loss，而不是对loss
        optim.step()  # 对每个参数进行调优
        running_loss = running_loss + result_loss  # 每一轮所有loss的和
    print(running_loss)
```
