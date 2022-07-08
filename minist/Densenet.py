import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision

from torch.nn import functional as F
from dataload import train_loader,test_loader
from trainer import train

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
    
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 10, trainer)

'''
1 (0.002827493621222675, 0.9391166666666667, 0.9822333333333333)
2 (0.0005070061471468459, 0.9866833333333334, 0.9850333333333333)
3 (0.0003448333618386338, 0.9904833333333334, 0.9795333333333334)
4 (0.0002841048375160123, 0.9922, 0.9902833333333333)
5 (0.0002384124694566708, 0.99335, 0.9941166666666666)
6 (0.0001972855575499125, 0.99485, 0.9951166666666666)
7 (0.00017382845449416588, 0.9951833333333333, 0.9935833333333334)
8 (0.00016259980264100401, 0.99535, 0.9952833333333333)
9 (0.00014597438611575246, 0.9960833333333333, 0.9940833333333333)
10 (0.00012647803993798637, 0.9964, 0.9979166666666667)
'''