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