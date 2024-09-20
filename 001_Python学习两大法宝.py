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