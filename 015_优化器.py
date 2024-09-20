import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Sequential, Conv2d, Flatten, MaxPool2d, Linear

dataset = torchvision.datasets.CIFAR10("related_data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(), 
            Linear(1024, 64),
            Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return(x)

loss = nn.CrossEntropyLoss()
mynetwork = MyNetwork()
optim = torch.optim.SGD(mynetwork.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0

    for data in dataloader:
        imgs, targets = data
        outputs = mynetwork(imgs)
        
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    
    print(running_loss)
