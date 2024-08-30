# PyTorch

**Github**：https://github.com/IvenStarry  
**学习视频网站**：  
B站小土堆PyTorch https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=6fd71d34326a08965fdb07842b0124a7


## 优化器
当使用损失函数时，可以调用损失函数的 backward，得到反向传播，反向传播可以求出每个需要调节的参数对应的梯度，有了梯度就可以利用优化器，优化器根据梯度对参数进行调整，以达到整体误差降低的目的
```python
# （1）构造
# Example：
# SGD为构造优化器的算法，Stochastic Gradient Descent 随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  #模型参数、学习速率、特定优化器算法中需要设定的参数
optimizer = optim.Adam([var1, var2], lr=0.0001)

# （2）调用优化器的step方法
for input, target in dataset:
    optimizer.zero_grad() #把上一步训练的每个参数的梯度清零
    output = model(input)
    loss = loss_fn(output, target)  # 输出跟真实的target计算loss
    loss.backward() #调用反向传播得到每个要更新参数的梯度
    optimizer.step() #每个参数根据上一步得到的梯度进行优化
```
学习速率不能太大（太大模型训练不稳定）也不能太小（太小模型训练慢），一般建议先采用较大学习速率，后采用较小学习速率
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

## 现有网络模型的使用及修改
![alt text](image.png)
![alt text](image-1.png)
ImageNet 数据集
![alt text](image-2.png)
VGG 模型
![alt text](image-3.png)
- pretrained 为 False 的情况下，只是加载网络模型，参数都为默认参数，不需要下载
- 为 True 时需要从网络中下载，卷积层、池化层对应的参数等等（在ImageNet数据集中训练好的）

```python
# vgg网络结构
import torchvision.models
 
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 另一个参数progress显示进度条，默认为True
vgg16_true = torchvision.models.vgg16(pretrained=True)
 
print(vgg16_true)
```
**如何利用现有网络去改动它的结构？**
train_data = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
- 把最后线性层的 out_features 从1000改为10
- 在最后的线性层下面再加一层，in_features为1000，out_features为10

```python
# 添加
# 给 vgg16 添加一个线性层，输入1000个类别，输出10个类别
vgg16_true.add_module('add_linear',nn.Linear(in_features=1000,out_features=10))
print(vgg16_true)

# 给 vgg16 添加一个线性层，输入1000个类别，输出10个类别
vgg16_true.classifier.add_module('add_linear',nn.Linear(in_features=1000,out_features=10))
print(vgg16_true)

# 修改
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 另一个参数progress显示进度条，默认为True
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
```

## 网络模型的保存与读取
![alt text](image-4.png)
**两种方式保存模型**
- 方式1：不仅保存了网络模型的结构，也保存了网络模型的参数
- 方式2：网络模型的参数保存为字典，不保存网络模型的结构（官方推荐的保存方式，用的空间小）

```python
# 保存方式1：模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2：模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")   
# 把vgg16的状态保存为字典形式（Python中的一种数据格式）
```

**两种方式加载模型**
- 方式1：对应保存方式1，打印出的是网络模型的结构 
- 方式2：对应保存方式2，打印出的是参数的字典形式

```python
# 方式1 对应 保存方式1，加载模型
model = torch.load("vgg16_method1.pth",)
print(model)  # 打印出的只是模型的结构，其实它的参数也被保存下来了

# 方式2 加载模型
model = torch.load("vgg16_method2.pth")
print(model)
```

如何恢复网络模型结构？
```python
import torchvision.models
 
vgg16 = torchvision.models.vgg16(pretrained=False)  # 预训练设置为False
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # vgg16通过字典形式，加载状态即参数
print(vgg16)
```

`model.py`
```python
# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv = nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):   # x为输入
        x = self.conv(x)
        return x
    
model = torch.load("tudui_method1.pth")
print(model)
```

## 完整的模型训练套路
![alt text](image-5.png)
`model.py`
```python
import torch
from torch import nn
 
# 搭建神经网络（10分类网络）
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 把网络放到序列中
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2), #输入是32x32的，输出还是32x32的（padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),  #输入输出都是16x16的（同理padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),   # 展平
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        x = self.model(x)
        return x
 
if __name__ == '__main__':
    # 测试网络的验证正确性
    tudui = Tudui()
    input = torch.ones((64,3,32,32))  # batch_size=64（代表64张图片）,3通道，32x32
    output = tudui(input)
    print(output.shape)
```
`train.py`
```python
import torchvision.datasets
from model import *
from torch import nn
from torch.utils.data import DataLoader
 
# 准备数据集，CIFAR10 数据集是PIL Image，要转换为tensor数据类型
train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
 
# 看一下训练数据集和测试数据集都有多少张（如何获得数据集的长度）
train_data_size = len(train_data)   # length 长度
test_data_size = len(test_data)
# 如果train_data_size=10，那么打印出的字符串为：训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))   # 字符串格式化，把format中的变量替换{}
print("测试数据集的长度为：{}".format(test_data_size))
 
# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
 
# 创建网络模型
tudui = Tudui()
 
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()   # 分类问题可以用交叉熵
 
# 定义优化器
learning_rate = 0.01   # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)   # SGD 随机梯度下降
 
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10   # 训练轮数
 
for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i+1))  # i从0-9
    # 训练步骤开始
    for data in train_dataloader:   # 从训练的dataloader中取数据
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
 
        # 优化器优化模型
        optimizer.zero_grad()    # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()   # 对参数进行优化
        total_train_step += 1
        print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
```

**如何知道模型是否训练好，或达到需求？**
每次训练完一轮就进行测试，以测试数据集上的损失或正确率来评估模型有没有训练好，测试过程中不需要对模型进行调优，利用现有模型进行测试.
```python
# 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字，所以要.item()一下
    print("整体测试集上的Loss：{}".format(total_test_loss))
```
与tensorboard结合应用
```python
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
 
from model import *
from torch import nn
from torch.utils.data import DataLoader
 
# 准备数据集，CIFAR10 数据集是PIL Image，要转换为tensor数据类型
train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
 
# 看一下训练数据集和测试数据集都有多少张（如何获得数据集的长度）
train_data_size = len(train_data)   # length 长度
test_data_size = len(test_data)
# 如果train_data_size=10，那么打印出的字符串为：训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))   # 字符串格式化，把format中的变量替换{}
print("测试数据集的长度为：{}".format(test_data_size))
 
# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
 
# 创建网络模型
tudui = Tudui()
 
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()   # 分类问题可以用交叉熵
 
# 定义优化器
learning_rate = 0.01   # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)   # SGD 随机梯度下降
 
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10   # 训练轮数
 
# 添加tensorboard
writer = SummaryWriter("../logs_train")
 
for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i+1))  # i从0-9
    # 训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
 
        # 优化器优化模型
        optimizer.zero_grad()    # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()   # 对参数进行优化
        total_train_step += 1
        if total_train_step % 100 ==0:  # 逢百才打印记录
            print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
 
    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
    print("整体测试集上的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    total_test_step += 1
 
writer.close()
```
![alt text](image-6.png)

**保存每一轮训练的模型**
```python
    torch.save(tudui,"tudui_{}.pth".format(i))  # 每一轮保存一个结果
    print("模型已保存")
 
writer.close()
```
**正确率的实现**（对分类问题） 
即便得到整体测试集上的 loss，也不能很好说明在测试集上的表现效果
- 在分类问题中可以用正确率表示（下述代码改进）
- 在目标检测/语义分割中，可以把输出放在tensorboard里显示，看测试结果

![alt text](image-7.png)
```python
import torch
 
outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])
print(outputs.argmax(1))  # 0或1表示方向，1为横向比较大小. 运行结果：tensor([1, 1])

preds = outputs.argmax(1)
targets = torch.tensor([0,1])
print(preds == targets)  # tensor([False,  True])
print(sum(preds == targets).sum())  # tensor(1)，对应位置相等的个数

# 计算整体正确率
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
            # 求整体测试数据集上的误差或正确率
            accuracy = (outputs.argmax(1) == targets).sum()  # 1：横向比较，==：True或False，sum：计算True或False个数
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/imgs.size(0)))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/imgs.size(0),total_test_step)
    total_test_step += 1
 
    torch.save(tudui,"tudui_{}.pth".format(i))  # 每一轮保存一个结果
    print("模型已保存")
 
writer.close()
```

**model.train() 和 model.eval()**
- 训练步骤开始之前会把网络模型（我们这里的网络模型叫 tudui）设置为train，并不是说把网络设置为训练模型它才能够开始训练
- 测试网络前写 网络.eval()，并不是说需要这一行才能把网络设置成 eval 状态，才能进行网络测试

![alt text](image-8.png)
![alt text](image-9.png)
```python
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
 
from model import *
from torch import nn
from torch.utils.data import DataLoader
 
# 准备数据集，CIFAR10 数据集是PIL Image，要转换为tensor数据类型
train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
 
# 看一下训练数据集和测试数据集都有多少张（如何获得数据集的长度）
train_data_size = len(train_data)   # length 长度
test_data_size = len(test_data)
# 如果train_data_size=10，那么打印出的字符串为：训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))   # 字符串格式化，把format中的变量替换{}
print("测试数据集的长度为：{}".format(test_data_size))
 
# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
tudui = Tudui()
 
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()   # 分类问题可以用交叉熵
 
# 定义优化器
learning_rate = 0.01   # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)   # SGD 随机梯度下降
 
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10   # 训练轮数

    # 训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
 
        # 优化器优化模型
        optimizer.zero_grad()    # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()   # 对参数进行优化
        total_train_step += 1
        if total_train_step % 100 ==0:  # 逢百才打印记录
            print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
            # 求整体测试数据集上的误差或正确率
            accuracy = (outputs.argmax(1) == targets).sum()  # 1：横向比较，==：True或False，sum：计算True或False个数
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step += 1
```

示例2
`train.py`
```python
import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
 
# from model import *
from torch import nn
from torch.utils.data import DataLoader
import time  # time这个package是用来计时的
 
# 准备数据集，CIFAR10 数据集是PIL Image，要转换为tensor数据类型
device = torch.device("cuda")
print(device)
train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
 
# 看一下训练数据集和测试数据集都有多少张（如何获得数据集的长度）
train_data_size = len(train_data)   # length 长度
test_data_size = len(test_data)
# 如果train_data_size=10，那么打印出的字符串为：训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))   # 字符串格式化，把format中的变量替换{}
print("测试数据集的长度为：{}".format(test_data_size))
 
# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
 
 
# 搭建神经网络（10分类网络）
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 把网络放到序列中
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2), #输入是32x32的，输出还是32x32的（padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),  #输入输出都是16x16的（同理padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),   # 展平
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        x = self.model(x)
        return x
 
# 创建网络模型
tudui = Tudui()
tudui.to(device)
 
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()   # 分类问题可以用交叉熵
loss_fn.to(device)
 
# 定义优化器
learning_rate = 0.01   # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)   # SGD 随机梯度下降
 
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 30   # 训练轮数
 
# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()  # 记录下此时的时间，赋值给开始时间
 
for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i+1))  # i从0-9
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
 
        # 优化器优化模型
        optimizer.zero_grad()    # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()   # 对参数进行优化
        total_train_step += 1
        if total_train_step % 100 ==0:  # 逢百才打印记录
            end_time = time.time()
            print(end_time - start_time)  # 第一次训练100次所花费的时间
            print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
 
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
            accuracy = (outputs.argmax(1) == targets).sum()  # 1：横向比较，==：True或False，sum：计算True或False个数
            total_accuracy = total_accuracy + accuracy
        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))  # 正确率为预测对的个数除以测试集长度
        writer.add_scalar("test_loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",total_test_loss,total_test_step,total_test_step)
        total_test_step += 1
 
        torch.save(tudui,"tudui_{}_gpu.pth".format(i))  # 每一轮保存一个结果
        print("模型已保存")
 
writer.close()
```
`test.py`
```python
# 如何从test.py文件去找到dog文件（相对路径）
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
 
image_path = "../imgs/airplane.png"  # 或右键-> Copy Path-> Absolute Path（绝对路径）
# 读取图片（PIL Image），再用ToTensor进行转换
image = Image.open(image_path)  # 现在的image是PIL类型
print(image)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=430x247 at 0x1DF29D33AF0>
 
# image = image.convert('RGB')
# 因为png格式是四通道，除了RGB三通道外，还有一个透明度通道，所以要调用上述语句保留其颜色通道
# 当然，如果图片本来就是三颜色通道，经过此操作，不变
# 加上这一步后，可以适应 png jpg 各种格式的图片
 
# 该image大小为430x247，网络模型的输入只能是32x32，进行一个Resize()
# Compose()：把transforms几个变换联立在一起
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),  # 32x32大小的PIL Image
                                            torchvision.transforms.ToTensor()])  # 转为Tensor数据类型
image = transform(image)
print(image.shape)  # torch.Size([3, 32, 32])
 
# 加载网络模型（之前采用的是第一种方式保存，故需要采用第一种方式加载）
# 首先搭建神经网络（10分类网络）
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 把网络放到序列中
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2), #输入是32x32的，输出还是32x32的（padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),  #输入输出都是16x16的（同理padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),   # 展平
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        x = self.model(x)
        return x
# 然后加载网络模型
model = torch.load("tudui_29_gpu.pth",map_location=torch.device('cpu'))  # 因为test.py和tudui_0.pth在同一个层级下，所以地址可以直接写
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()  # 模型转化为测试类型
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
```