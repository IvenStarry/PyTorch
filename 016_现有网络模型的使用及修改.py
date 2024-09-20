import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("related_data", split="train", download=True, transform=torchvision.transforms.ToTensor())

# 若pretrained设置为True，会使用在数据集上已经训练好的参数，若为False，则使用默认的初始化参数
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("related_data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# * path1：使用add_module() 添加新层
vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10)) # 将添加的线性层加在classifier中
print(vgg16_true)

# * path2：只修改当前层(第七层)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)