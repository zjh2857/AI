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
    channels = 1
).cuda()


criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.001372994857141748, 0.9564166666666667, 0.9865666666666667)
2 (0.00048043399758171293, 0.9848, 0.9923)
3 (0.0003185815174986298, 0.9899333333333333, 0.9952666666666666)
4 (0.0002336374769016402, 0.9924666666666667, 0.9952833333333333)
5 (0.00017482907654921292, 0.99485, 0.9973)
'''
