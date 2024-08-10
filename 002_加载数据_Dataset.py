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