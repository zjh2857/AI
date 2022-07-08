import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision

from torch.nn import functional as F
from dataload import train_loader,test_loader
from trainer import train

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10)).cuda()
        
criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
train(net, train_loader, test_loader, criterion, 10, trainer)

'''
1 (0.0019760091803657513, 0.9421, 0.97735)
2 (0.0005251034660769316, 0.9837833333333333, 0.98985)
3 (0.0003566122215997893, 0.9889, 0.9933166666666666)
4 (0.00026315733544761313, 0.9915, 0.9877333333333334)
5 (0.00021865047564497217, 0.9930166666666667, 0.9940666666666667)
6 (0.000180646648054244, 0.9944, 0.9966833333333334)
7 (0.0001471480121096344, 0.9951, 0.98475)
8 (0.00012414847095157408, 0.99575, 0.9978666666666667)
9 (0.00011290839550638339, 0.9963333333333333, 0.9985333333333334)
10 (8.271039825267507e-05, 0.9972666666666666, 0.9965166666666667)
'''