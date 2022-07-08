import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision


from dataload import train_loader,test_loader
from trainer import train

net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(3, 96, kernel_size=4, stride=2, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(256, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10)).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 10, trainer)



'''
1 (0.07659722643136978, 0.1794, 0.1724)
2 (0.05966403482198715, 0.30158, 0.3454)
3 (0.054488858389854435, 0.37304, 0.4069)
4 (0.050386551105976105, 0.41894, 0.4198)
5 (0.047208979489803314, 0.45806, 0.4449)
6 (0.04473426777601242, 0.48902, 0.4929)
7 (0.042838359602689745, 0.50956, 0.5225)
8 (0.041175957258939744, 0.52916, 0.5487)
9 (0.03955078942656517, 0.54638, 0.5444)
10 (0.038199941455125806, 0.56292, 0.5577)
'''