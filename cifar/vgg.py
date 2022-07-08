import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision

from torch.nn import functional as F

from dataload import train_loader,test_loader
from trainer import train

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
train(net, train_loader, test_loader, criterion, 10, trainer)


'''
1 (0.023015219600995382, 0.11173333333333334, 0.11236666666666667)
2 (0.022635713817675908, 0.1437, 0.15173333333333333)
3 (0.002667809399360946, 0.91565, 0.9827833333333333)
4 (0.0005568731328627715, 0.9830833333333333, 0.9905833333333334)

5 (0.0003800764682411682, 0.9884, 0.9933)
6 (0.0002857716536944887, 0.99125, 0.9948333333333333)
'''