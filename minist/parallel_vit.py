import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

from vit_pytorch.parallel_vit import ViT

net = ViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    num_parallel_branches = 2,  # in paper, they claimed 2 was optimal
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 1
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.03177613467276096, 0.22073333333333334, 0.6248166666666667)
2 (0.006127335976809263, 0.8013333333333333, 0.9150833333333334)
3 (0.002677109867706895, 0.9151833333333333, 0.95655)
4 (0.0017904593444739779, 0.9449666666666666, 0.9674166666666667)
5 (0.0013779006310893844, 0.95675, 0.9762166666666666)
'''