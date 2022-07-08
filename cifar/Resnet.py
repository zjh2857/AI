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
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
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
1 (0.05490040193915367, 0.3618, 0.0997)
2 (0.04047401192426681, 0.53294, 0.4587)
3 (0.0329564579230547, 0.62672, 0.5334)
4 (0.027688031000494957, 0.68872, 0.6674)
5 (0.023955845267772675, 0.73206, 0.6889)
6 (0.021119772936701774, 0.76512, 0.6586)
7 (0.01842633289426565, 0.79568, 0.7029)
8 (0.016029177816212176, 0.82088, 0.6604)
9 (0.0139229271620512, 0.84444, 0.6895)
10 (0.011949057186357677, 0.86758, 0.6549)
'''