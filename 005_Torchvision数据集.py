import torchvision
from torch.utils.tensorboard import SummaryWriter

'''
CIFAR10(root: Union[str, Path], train=True, transform=None, target_transform= None, download= False)
root:               数据集路径
train:              是否作为训练集，默认True，若为False则作为测试集
transform           预处理操作
target_transform    对标签进行预处理操作
download            自动下载数据集压缩包并解压(url地址可以在定义中查看)
'''

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='related_data', train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root='related_data', train=False, transform=dataset_transform, download=False)

print(test_set.classes) # 输出所有标签名
img, target = test_set[0] # 返回图像和标签序号
print(img)
print(target)
print(test_set.classes[target])
# img.show() # PIL image可以使用show方法展示图片 经过transform后转为Tensor格式没有show方法

print(test_set[0]) # 转为tensor格式后返回了一个矩阵和标签序号

writer = SummaryWriter('logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()