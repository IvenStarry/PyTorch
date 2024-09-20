from torch import nn
import torch

class My_Module(nn.Module):
    def __int__(self): # 在给子类分配任务前，先调用父类方法__init__()
        super().__init__()
    
    def forward(self, input): # froward: 定义每次调用时执行的计算，应被所有子类所覆盖
        output = input + 1
        return output

my_module = My_Module()
x = torch.tensor(1.0)
output = my_module(x)
print(output)