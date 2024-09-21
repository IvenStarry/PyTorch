import torch
import torchvision
from PIL import Image
from torch import nn

# 在当前目录下寻找文件 可写./ 或省略./直接写文件;若文件路径在上一层目录，可写../
image_path = "./related_data/airplane.jpg"
image = Image.open(image_path)
print(type(image))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)
print(type(image))

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# * 如果使用GPU训练的模型,使用map_location对应到cpu上
model = torch.load("./related_data/mynetwork_018_9.pth", map_location=torch.device("cpu"))
print(model)

image = torch.reshape(image, (1, 3, 32, -1)) # 网络训练要求有batch_size reshape成对应的形状
model.eval()
with torch.no_grad(): # 可以节约性能
    output = model(image)
print(output)

print(output.argmax(1))
print()