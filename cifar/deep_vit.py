import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

net = DeepViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 3
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 10, trainer)
print("aaa")

'''
1 (0.08063363995313644, 0.18432, 0.244)
2 (0.05882051604032516, 0.30128, 0.3471)
3 (0.05442474078655243, 0.36098, 0.3407)
4 (0.0514549213719368, 0.40148, 0.4007)
5 (0.04832661584258079, 0.44162, 0.4714)
6 (0.045532042752504345, 0.47402, 0.4875)
7 (0.04325389968395233, 0.50314, 0.4869)
8 (0.04129096463561058, 0.52576, 0.5176)
9 (0.039696285078525545, 0.5442, 0.5206)
10 (0.03813777401566505, 0.56462, 0.5178)
'''