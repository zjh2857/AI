import torch
from vit_pytorch.deepvit import DeepViT
from torch import nn
from dataload import train_loader,test_loader
from trainer import train

net = DeepViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1
).cuda()

criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
train(net, train_loader, test_loader, criterion, 5, trainer)
print("aaa")

'''
1 (0.034404549459616345, 0.1068, 0.10656666666666667)
2 (0.020592023491859436, 0.22798333333333334, 0.5172166666666667)
3 (0.008129453605165085, 0.7154, 0.7270166666666666)
4 (0.003257044892758131, 0.8972166666666667, 0.9162)
5 (0.002138842036947608, 0.9332, 0.92145)
'''