import torchvision
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("related_data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("related_data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
mynetwork = MyNetwork()
# ! 将网络模型转移到cuda上来
if torch.cuda.is_available():
    mynetwork = mynetwork.cuda()

loss_function = nn.CrossEntropyLoss()
# ! 将损失函数转移到cuda上来
loss_function = loss_function.cuda()
learning_rate = 1e-2
optimizer = torch.optim.SGD(mynetwork.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10
start_time = time.time()

for i in range(epoch):
    print(f"----------第 {i} 轮训练开始----------")

    mynetwork.train()
    for data in train_dataloader:
        imgs, targets = data
        # ! 数据（样本和标签） 转移到cuda上来
        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = mynetwork(imgs)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("前100轮使用时间 %.10f" % (end_time - start_time))
            print(f"训练次数：{total_train_step}, Loss: {loss}")
    
    mynetwork.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # ! 数据（样本和标签） 转移到cuda上来
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = mynetwork(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集的正确率: {total_accuracy / test_data_size}")
    total_test_step += 1

    torch.save(mynetwork, "related_data/GPU_MyNetwork_{}".format(i))
    print(f"模型已保存")
