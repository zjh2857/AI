import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision

trans_to_tensor = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor()
])

data_train = torchvision.datasets.MNIST(
    './data', 
    train=True, 
    transform=trans_to_tensor, 
    download=True)

data_test = torchvision.datasets.MNIST(
    './data', 
    train=False, 
    transform=trans_to_tensor, 
    download=True)


train_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=False)