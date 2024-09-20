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