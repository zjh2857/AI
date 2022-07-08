import torch
from vit_pytorch import ViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

net = ViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels = 1
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.0327637209157149, 0.35231666666666667, 0.77275)
2 (0.004626639473438263, 0.8539, 0.9164666666666667)
3 (0.002510160459702214, 0.9205, 0.95545)
4 (0.0017986671078329285, 0.9433833333333334, 0.9631833333333333)
5 (0.0014309665576865275, 0.9562666666666667, 0.9699666666666666)
'''