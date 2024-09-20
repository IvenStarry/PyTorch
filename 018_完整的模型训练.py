import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter

# 获取数据集
train_data = torchvision.datasets.CIFAR10("related_data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("related_data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print(f"测试数据集的长度为：{test_data_size}")

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建网络模型并生成对象
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
mynetwork = MyNetwork()

# 输出形状测试
# if __name__ == '__main__':
#     mynetwork = MyNetwork()
#     input = torch.ones(64, 3, 32, 32) # 可以想象是64张3通道的(32,32)图像
#     output = mynetwork(input)
#     # print(output.shape)

# 损失函数与优化器
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-2 #  0.01
optimizer = torch.optim.SGD(mynetwork.parameters(), lr=learning_rate)

# 记录训练的次数
total_training_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加Tensorboard
writer = SummaryWriter("logs_network")

for i in range(epoch):
    print(f"---------第 {i + 1} 轮训练开始---------")

    # 训练开始
    mynetwork.train() # 只对Dropout层、BatchNorm层有作用
    for data in train_dataloader:
        imgs, targets = data
        outputs = mynetwork(imgs) # 得到输出

        loss = loss_function(outputs, targets) # loss计算
        optimizer.zero_grad() # 梯度清0
        loss.backward() # 反向传播
        optimizer.step() # 参数优化

        total_training_step += 1
        if total_training_step % 100 == 0:
            print("训练次数:{}, loss:{}".format(total_training_step, loss)) # 不加item会显示loss的数据类型(Tensor[])，加item只显示数值
            writer.add_scalar("Training Loss", loss.item(), total_training_step) # 绘制折线图
    
    # 测试开始
    mynetwork.eval() # 只对Dropout层、BatchNorm层有作用
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad(): # 设置无梯度
            for data in test_dataloader:
                imgs, targets = data
                outputs = mynetwork(imgs)

                loss = loss_function(outputs, targets)
                total_test_loss += loss
                accuracy = (outputs.argmax(1) == targets).sum() # 正确率计算 test在最后
                total_accuracy += accuracy
    
    # 显示测试集的正确率与Loss
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：%f" % (total_accuracy / test_data_size))
    writer.add_scalar("Test Loss", total_test_loss.item(), total_test_step) # 绘制折线图
    writer.add_scalar("Test Accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(mynetwork, "related_data/mynetwork_018_{}.pth".format(i))
    print("模型mynetwork_018_{}已保存".format(i))

writer.close()

# 正确率计算测试
# outputs = torch.tensor([[0.1, 0.2],
#                         [0.05, 0.4]])

# # argmax(dim) 返回指定维度最大值的序号 dim：指定观察的维度
# print(outputs.argmax(0))
# print(outputs.argmax(1))

# preds = outputs.argmax(1)
# targets = torch.tensor([0, 1])
# print("preds:%s" % preds)
# print("targets:%s" % targets)
# print(preds == targets)
# print((preds == targets).sum()) # sum输出对应位置相等的的个数