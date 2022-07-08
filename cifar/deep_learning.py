import numpy as np
import torch 
from torch.utils import data
from torch import nn

from torchvision import transforms
import torchvision
from dataload import train_loader,test_loader
from trainer import train


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.conv = nn.Conv2d(3,1,kernel_size=1)
        self.GRU_layer = nn.GRU(input_num,hidden_num,batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1,32,32)
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x[:,-1,:])
        return x

class LSTMModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_num
        self.conv = nn.Conv2d(3,1,kernel_size=1)
        self.GRU_layer = nn.LSTM(input_num,hidden_num,batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1,32,32)
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x[:,-1,:])
        return x

class RNNModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_num
        self.conv = nn.Conv2d(3,1,kernel_size=1)
        self.GRU_layer = nn.RNN(input_num,hidden_num,batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1,32,32)
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x[:,-1,:])
        return x

# trans_to_tensor = transforms.Compose([
#     transforms.ToTensor()
# ])

# data_train = torchvision.datasets.MNIST(
#     './data', 
#     train=True, 
#     transform=trans_to_tensor, 
#     download=True)

# data_test = torchvision.datasets.MNIST(
#     './data', 
#     train=False, 
#     transform=trans_to_tensor, 
#     download=True)


# train_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=True)

# def test(net,reshape):
#     net.eval()
    
#     # test_loader = torch.utils.data.DataLoader(data_train, batch_size=10000, shuffle=False)
#     test_data = next(iter(test_loader))
    
#     with torch.no_grad():
#         x, y = test_data[0], test_data[1]
#         x = x.cuda()
#         y = y.cuda()
#         if(reshape):
#             x = x.view(-1,28,28)
#         outputs = net(x)
#         pred = torch.max(outputs, 1)[1]
#         print(f'test acc: {sum(pred == y) / outputs.shape[0]}')
    
#     net.train()

# def fit(net,epoch=10,reshape=False):
#     net.train()
#     run_loss = 0
#     for num_epoch in range(epoch):
#         print(f"run epoch{num_epoch + 1}")
#         for i,data in enumerate(train_loader):
#             X , y = data[0], data[1]
#             X = X.cuda()
#             y = y.cuda()
#             if(reshape):
#                 X = X.view(-1,28,28)
#             output = net(X)
#             loss = criterion(output,y)
#             trainer.zero_grad()
#             loss.backward()
#             trainer.step()
#             run_loss = loss.item()
#             if i % 100 == 99:
#                 print(f'[{(i+1) * 100} / 60000] loss={run_loss / 100}')
#                 run_loss = 0
#                 test(net,reshape)

# lr = 0.5
'''
LMS
'''
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072,10)
).cuda()
criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
train(net, train_loader, test_loader, criterion, 5, trainer)
# fit(net)
print("LMS")
# '''
# MLP
# '''

# net = nn.Sequential(nn.Flatten(),
#                     nn.Linear(3072, 256),
#                     nn.ReLU(),
#                     nn.Linear(256, 10)).cuda()
# criterion  = nn.CrossEntropyLoss()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# train(net, train_loader, test_loader, criterion, 10, trainer)

# # fit(net)
# print("MLP")
'''
DNN
'''
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(3072, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 4028),
                    nn.ReLU(),
                    nn.Linear(4028, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)).cuda()
criterion  = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

train(net, train_loader, test_loader, criterion, 5, trainer)
print("DNN")

# '''
# CNN
# '''
# net = nn.Sequential(
#     nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(576, 10)
#     ).cuda()
# criterion  = nn.CrossEntropyLoss()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# train(net, train_loader, test_loader, criterion, 5, trainer)

# print("CNN")
# '''
# RNN
# '''
# net = RNNModel(32,50,10).cuda()
# criterion  = nn.CrossEntropyLoss()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# train(net, train_loader, test_loader, criterion, 5, trainer)

# print("RNN")
# '''
# GRU
# '''
# net = GRUModel(32,50,10).cuda()
# criterion  = nn.CrossEntropyLoss()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# train(net, train_loader, test_loader, criterion, 5, trainer)

# print("GRU")
# '''
# LSTM
# '''
# net = LSTMModel(32,50,10).cuda()
# criterion  = nn.CrossEntropyLoss()
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# train(net, train_loader, test_loader, criterion, 5, trainer)
# print("LSTM")