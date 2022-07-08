import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

from vit_pytorch.learnable_memory_vit import ViT, Adapter

# normal base ViT

net = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 3
).cuda()


criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 10, trainer)
print("aaa")

'''
1 (0.0789397307920456, 0.18414, 0.2914)
2 (0.05793344731330872, 0.32542, 0.3749)
3 (0.053112448301315306, 0.39032, 0.3546)
4 (0.04938223090171814, 0.43272, 0.4818)
5 (0.04659111193537712, 0.46614, 0.4924)
'''