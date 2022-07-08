import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

from vit_pytorch.t2t import T2TViT

net = T2TViT(
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)), # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    channels = 1
).cuda()


criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.006699332921144863, 0.8220666666666666, 0.9566833333333333)
2 (0.0011559598695486783, 0.9655, 0.97595)
3 (0.0006981687837978825, 0.97885, 0.9849)
4 (0.0004778759445102575, 0.9855, 0.9892)
5 (0.00034696266532797986, 0.9892166666666666, 0.9924833333333334)
'''