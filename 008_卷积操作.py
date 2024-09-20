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