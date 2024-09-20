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