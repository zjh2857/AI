import torch
from vit_pytorch import ViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

net = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 3
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 100, trainer)
print("aaa")

'''
1 (0.08727959962129593, 0.17018, 0.1844)
2 (0.06166619770288467, 0.27352, 0.3381)
3 (0.055520512928962706, 0.35778, 0.2864)
4 (0.05177397869110108, 0.40546, 0.4259)
5 (0.04892272763252258, 0.43752, 0.4555)
'''