import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

from vit_pytorch.t2t import T2TViT

net = T2TViT(
    dim = 512,
    image_size = 32,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)), # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    channels = 3
).cuda()


criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.06153365221977234, 0.32278, 0.4154)
2 (0.04277196702837944, 0.5037, 0.4961)
3 (0.03697654676795006, 0.57332, 0.5525)
4 (0.032245898274183275, 0.6303, 0.5961)
5 (0.028149651042222976, 0.67614, 0.6081)
'''