import torch
from vit_pytorch import ViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

net = ViT(
    image_size = (2,39),
    patch_size = 1,
    num_classes = 15,
    dim = 1024,
    depth = 15,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 1
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("zzz")

'''
1 (0.005915227214022499, 0.8807830311497793, 0.7404515814221709)
2 (0.0010823089508697292, 0.9724621344425746, 0.7416049327951026)
3 (0.0007856972094546454, 0.9791540977360048, 0.7447988289047598)
4 (0.000705829844308817, 0.9811466040072397, 0.7700394801046888)
5 (0.0006561628816994812, 0.9818848633029562, 0.7625870558488222)
'''
