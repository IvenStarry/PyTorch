import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
DataLoader(dataset, batch_size=1, shuffle=None, num_workers=0, drop_last=False)
dataset:        数据存放位置
batch_size:     每次加载几个样本
shuffle:        若为True，则将整个数据集打乱顺序
num_worker:     采用多进程还是单进程加载数据，多进程加载更快，默认0将数据加载进主进程
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