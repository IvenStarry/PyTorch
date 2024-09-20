import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1(保存网络模型结构和参数)
torch.save(vgg16, "related_data/vgg16_method1.pth")

# 加载模型1
model = torch.load("related_data/vgg16_method1.pth")
print(model)

# * 加载模型1的陷阱

model = torch.load("related_data/MyNetwork.pth")
'''
AttributeError: Can't get attribute 'MyNetwork' on <module '__main__'
VSCode无法找到MyNetwork这个类的结构，故报上述错误
解决方法1：将类的这一部分代码放在加载模型之前(无需创建对象)
解决方法2：导入对应的类 使用from module_save import MyNetwork
'''

# 保存方式2(保存为字典结构，推荐)
torch.save(vgg16.state_dict(), "related_data/vgg16_method2.pth")

# 加载模型2并还原网络结构
model = torch.load("related_data/vgg16_method2.pth")
vgg16_method2 = torchvision.models.vgg16(weights=None)
vgg16_method2.load_state_dict(model)
print(vgg16_method2)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=64)

    def forward(self, x):
        x = self.conv1(x)
        return x

mynetwork = MyNetwork()
torch.save(mynetwork, "related_data/MyNetwork.pth")
