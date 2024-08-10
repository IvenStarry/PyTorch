# PyTorch

**Github**：https://github.com/IvenStarry  
**学习视频网站**：  
B站小土堆PyTorch https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=6fd71d34326a08965fdb07842b0124a7

## Python学习两大法宝
```python
import torch
print(torch.cuda.is_available())

# dir(): 打开。看见
print('----------------dir(torch)---------------------')
print(dir(torch))
print('----------------dir(torch.cuda)---------------------')
print(dir(torch.cuda))
print('----------------dir(torch.cuda.is_available)---------------------')
print(dir(torch.cuda.is_available))

# help(): 说明书
print('----------------help(torch.cuda.is_available)---------------------')
print(help(torch.cuda.is_available))
```

## PyTorch加载数据
Dataset与Dataloader对比
- Dataset 提供一种方式去获取数据及其label
- Dataloader 为后面的网络提供不同的数据形式
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408090938970.png)

常用**数据集**两种**形式**
- 文件夹的名称是它的label。
- label为文本格式，文本名称为图片名称，文本中的内容为对应的label。

## 加载数据_Dataset
```python
from torch.utils.data import Dataset
from PIL import Image
import os

# print(help(Dataset))

# os文件路径处理
root_dir = 'related_data/hymenoptera_data/train'
label_dir = 'ants'
path = os.path.join(root_dir, label_dir) # 路径拼接

# os获取文件路径下所有文件名 文件排序后由索引调用 000 001 002 003...
dir_path = 'related_data/hymenoptera_data/train/ants'
img_path_list = os.listdir(dir_path)
print(img_path_list[0]) # 输出第一个图片名 对应索引0

# PIL加载图片
img_path = 'related_data/hymenoptera_data/train/ants/69639610_95e0de17aa.jpg'
img = Image.open(img_path)
# img.show()

# Dataset 提供一种方式去获取数据及其label
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path) # 获取所有图片名称列表

    def __getitem__(self, idx): # 通过[]去访问 输入一个索引值key完成后续操作，一般用于迭代序列
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 获取一张图片的位置
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path) # 返回文件路径下数据集数量

root_dir = 'related_data/hymenoptera_data/train'
ants_label_dir = 'ants'
ants_dataset = MyData(root_dir, ants_label_dir)
print(ants_dataset[0]) # 给定图片索引0 返回两个值 图片和标签
img, label = ants_dataset[0]
img.show()

bees_label_dir = 'bees'
bees_dataset = MyData(root_dir, bees_label_dir)
print(bees_dataset[1]) # 给定图片索引1 返回两个值 图片和标签
img, label = bees_dataset[1]
img.show()

train_dataset = ants_dataset + bees_dataset # 数据集拼接
print(len(train_dataset))
print(len(ants_dataset))
print(len(bees_dataset))
img, label = train_dataset[123] # 索引123是蚂蚁数据集最后一个索引
img.show()
img, label = train_dataset[124] # 索引124是蜜蜂数据集第一个索引
img.show()

# 练习：样本标签在文本文件里，文件名称为图片名
root_dir = 'related_data/ants_bees_test/train'
target_dir = 'ants_image'
img_path = os.path.join(root_dir, target_dir)
label = target_dir.split('_')[0]
out_dir = 'ants_label'
for i in img_path:
    file_name = i.split('.jpg')[0] # 获取图片名，对应标签文本文件名称
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label) # 写入标签
```

## Tensorboard的使用
**Tensorboard**用于查看loss是否按照我们预想的变化，或者查看训练到某一步输出的图像是什么样
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408091119645.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408091141858.png)
```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2
import numpy as np

# tensorboard用于查看loss是否按照我们预想的变化，或者查看训练到某一步输出的图像是什么样
# print(help(SummaryWriter))

# 创建对象，传入事件文件保存目录
writer = SummaryWriter("logs")

'''
add_scalar()三个重要参数：
- tag：图像标题
- scalar_value：y值
- global_step：步长(x值)
'''
# 绘制图像 y = x
for i in range(100):
    writer.add_scalar("y = 2x", 2 * i, i)

'''
在终端输入
tensorboard --logdir=D:/test/logs tensorboard --logdir=logs 获得绘图地址
tensorboard --logdir=logs --port=6008  避免服务器拥挤可以更换默认端口
'''

'''
add_image()四个重要参数：
- tag：图像标题
- img_tensor：图像数据，仅支持torch.Tensor, numpy.ndarray, or string/blobname格式
- global_step: 步长(x值)
- dataformats：数据格式规范，默认CHW，可以选择CHW, HWC, HW, WH, etc.
'''

img_path = 'related_data/tensorboard_test/train/ants_image/0013035.jpg'
# path1: 使用Image库读取，Image转ndarray格式
img = Image.open(img_path)
print(type(img)) # add_image() 不接受Image类型的图像
img_array = np.array(img) # 转numpy
print(type(img_array))
print(img_array.shape)
# path2: 使用OpenCV读取，直接就是adarray格式
img_opencv = cv2.imread(img_path)
print(type(img_opencv))
print(img_opencv.shape)

writer = SummaryWriter("logs")
img_path = 'related_data/tensorboard_test/train/ants_image/0013035.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image('test', img_array, 1, dataformats='HWC') # 跟默认图像形状不一致，指定数据格式HWC

img_path = 'related_data/tensorboard_test/train/bees_image/16838648_415acd9e3f.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image('test', img_array, 2, dataformats='HWC') # title未变因此可以直接添加不同step时的图像

writer.close()

'''
发生异常: TypeError
Cannot handle this data type: (1, 1, 512), |u1
默认输入图像格式C,H,W  因为图像原形状为 H,W,C 因此需要指定dataformats
'''
```

## Transforms的使用
**transforms工具箱**：给定特定格式的图片，经过工具处理，得到新的图片(预处理操作)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408091721054.png)
> 工具箱的使用：关注输入和输出类型，多看官方文档，关注方法需要什么参数
返回值类型查看
- print
- print(type())
- debug 添加断点 在变量中查看
```python
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter

'''
transforms工具箱: 给定特定格式的图片，经过工具处理，得到新的图片(预处理操作)
Tensor 张量 有一些属性，比如反向传播、梯度等属性，它包装了神经网络需要的一些属性
'''

# 通过transform.ToTensor 
img_path = 'related_data/hymenoptera_data/train/ants/0013035.jpg'
img_path_abs = 'D:/Coding/PyTorch/related_data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path)
print(img)
cv_img = cv2.imread(img_path)
print(type(cv_img))

# 调用对象
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
print(type(tensor_img))

writer = SummaryWriter("logs")
writer.add_image('Tensor_img', tensor_img)
writer.close()

# todo 常用的Transforms工具
# * 1. Compose() 传入一个操作列表，将多个图像预处理操作组合在一起，以便在深度学习任务中按顺序应用这些操作 通过魔术方法__call__实现
# __call__魔术方法 将一个类变成一个函数（使这个类的实例可以像函数一样调用） x() 与 x.__call__() 是相同的
class Person:
    def __call__(self, name):
        print('__call__' + 'Hello' + name)
    
    def hello(self, name):
        print('hello' + name)

person = Person()
person('Iven') # person成了函数 还可以传入参数
person.hello('Rosennn')

# * 2. ToTensor() 输入PIL或nadarray格式的图像，转换为tensor类型
img = Image.open('related_data/dog.jpg')
print(img)
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer = SummaryWriter("logs")
writer.add_image('ToTensor', img_tensor)


# * 3. ToPILImage() 输入ndarray或tensor格式的图像，转换为PIL类型

# * 4. Normalize() 输入三通道均值、三通道标准差，对tensor格式图像执行归一化操作 output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm)

# * 5. Resize() 将PIL图像调整至给定尺寸大小，size是序列(h, w)则调整至指定大小，输入一个int类型，将图片短边缩放至int，长宽比保持不变
print(img.size)
# path1 一步步执行操作
trans_resize = transforms.Resize((256, 256)) # PIL -> PIL
img_resize = trans_resize(img) # PIL -> Tensor
img_resize = trans_totensor(img_resize)
print(img_resize.size)
writer.add_image('Resize', img_resize, 0)
# path2 利用Compose顺序执行操作
trans_resize_2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

# * 6. RomdomCrop() 在随机位置裁剪给定的图像 size是序列(h, w)则随机裁剪至指定大小，输入一个int类型，将图片随机裁剪至正方形(int, int)
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)

writer.close()
```

## Torchvision数据集
```python
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
```

## 加载数据_DataLoader
**Dataset**与**DataLoader**使用魔术方法__getitem()__**返回值**的区别:
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202408102347590.png)
```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
DataLoader(dataset, batch_size=1, shuffle=None, num_workers=0, drop_last=False)
dataset:        数据存放位置
batch_size:     每次加载几个样本
shuffle:        若为True，则将整个数据集打乱顺序
num_workers:     采用多进程还是单进程加载数据，多进程加载更快，默认0将数据加载进主进程
drop_last:      如果数据集大小不能被批大小整除，则设置为True以删除最终多余的数据。如果为False，若数据集有多余，则最后一批将更小
'''

# 准备测试集数据
test_data = torchvision.datasets.CIFAR10('related_data', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试集第一图像和target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs')
step = 0
# 将每个batch中的图片及标签分别打包
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images('test_data', imgs, step)
    step += 1

# shuffle 每次读取(抓取)DataLoader时会打乱数据集顺序
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch:{}'.format(epoch), imgs, step)
        step += 1

writer.close()
# 注意drop_last为False，即剩余数据集保留，最后一个批次数量更少
'''
错误记录：tensorboard step显示不全，图像显示step跳步
错误原因：--samples_per_plugin 参数用来设置各个插件显示的最大数据样本数。例如，你可以设置标量、直方图、图像等插件的不同样本显示数。
        默认情况下，TensorBoard 可能不会显示所有记录的数据样本，特别是当数据量很大时。这样可以防止浏览器因为加载过多数据而崩溃。
解决办法：tensorboard --logdir=logs --samples_per_plugin=images=1000
'''
```
