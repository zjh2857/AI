from torch.utils.data import DataLoader,Dataset
import os 
import pickle
import numpy as np 
from PIL import Image
from torchvision import transforms

class CIFAR10(Dataset):
    def __init__(self, root, transform=None, target_transform=None,train=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_list = ['data_batch_1','data_batch_2','data_batch_3',
                    'data_batch_4','data_batch_5']
        self.test_list = ['test_batch']

        self.data = []
        self.targets = []
 
        if(self.train):
            choose_list=self.train_list
        else:
            choose_list=self.test_list
 
        for file_name in choose_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')#解析py2存储的文件
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)#vstack它是垂直（按照行顺序）的把数组给堆叠起来
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC，维度转换
 
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
 
        if self.transform is not None:
            img = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return img, target
 
    def __len__(self):
        return len(self.data)

root = 'cifar-10-batches-py'
cifar_train = CIFAR10(root,transform = transforms.Compose([
    transforms.Resize((32, 32)),transforms.ToTensor()]))
cifar_test = CIFAR10(root,transform = transforms.Compose([
    transforms.Resize((32, 32)),transforms.ToTensor()]),train=False)
batchsz = 32
train_loader = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(cifar_test,batch_size = batchsz,shuffle=True)
