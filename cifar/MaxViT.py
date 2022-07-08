import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train
from vit_pytorch.max_vit import MaxViT

net = MaxViT(
    num_classes = 10,
    dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
    dim = 96,                         # dimension of first layer, doubles every layer
    dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
    depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
    window_size = 1,                  # window size for block and grids
    mbconv_expansion_rate = 4,        # expansion rate of MBConv
    mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
    dropout = 0.1,                     # dropout
    channels = 3
).cuda()


criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 10, trainer)
print("aaa")

'''
1 (0.055923137608766556, 0.3957, 0.4361)
2 (0.04048568319439888, 0.56024, 0.5082)
3 (0.031978215465545655, 0.65122, 0.6572)
4 (0.025620267209112643, 0.7169, 0.6872)
5 (0.021716535752415656, 0.76206, 0.7302)
6 (0.018858017739653586, 0.79042, 0.7276)
7 (0.01665809278652072, 0.81714, 0.7524)
8 (0.014712134796380996, 0.8375, 0.7554)
9 (0.012901260293722152, 0.85666, 0.7614)
10 (0.011406282220110297, 0.87474, 0.7772)
'''
