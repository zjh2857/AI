import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

from vit_pytorch.learnable_memory_vit import ViT, Adapter

# normal base ViT

net = ViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 8,
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
1 (0.02130135650485754, 0.41973333333333335, 0.8409333333333333)
2 (0.0036664807396630445, 0.8834166666666666, 0.9396333333333333)
3 (0.002167351249915858, 0.9314833333333333, 0.9557666666666667)
4 (0.0016686839140020312, 0.9483833333333334, 0.963)
5 (0.0014136704969219863, 0.9561, 0.9732)
'''