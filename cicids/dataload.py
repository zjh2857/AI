import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib as matplot
import matplotlib.pyplot as plt
# %matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import torch
from torch.utils.data import Dataset

import warnings, os 
# warnings.filterwarnings("ignore")

# from keras import Sequential
# from keras.models import Model, load_model
# from keras.layers import *
# from keras.callbacks import ModelCheckpoint
# from keras import regularizers

from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, TruncatedSVD, PCA
from sklearn.svm import LinearSVC

# ls = [] 
# for filename in os.listdir(r'./MachineLearningCVE/'):
#   if '.csv' in  filename:
#     print(filename)
#     df = pd.read_csv("./MachineLearningCVE/" + filename)
#     ls.append(df)
#     print(f'Shape: {df.shape}. Attack Type {df[" Label"].unique()}')

# for df in ls:
#   cols = df.columns.to_list()
#   for i in range(len(cols)):
#     cols[i] = cols[i].strip()
#   df.columns = cols

# df = pd.concat(ls)

# df = pd.concat([df[df['Label'] != 'BENIGN'], df[df['Label'] == 'BENIGN'].sample(frac=.1, random_state=42)]) 

# le = LabelEncoder()
# df['Label'] = le.fit_transform(df['Label'])


# df.dropna(inplace=True)
# indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
# df = df[indices_to_keep]

# for i in df.columns:
#     df = df[df[i] != "Infinity"]
#     df = df[df[i] != np.nan]
#     df = df[df[i] != np.inf]
#     df = df[df[i] != -np.inf]
#     df = df[df[i] != ",,"]
#     df = df[df[i] != ", ,"]

import pickle
f = open("dump","rb")

# pickle.dump(df,f,0)
df = pickle.load(f)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Label'],axis=1), df['Label'], test_size=.20, random_state=42)
print(len(X_train))
class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self, X,y):
        self.X = torch.tensor(np.array(X)).to(torch.float32)
        self.y = torch.tensor(np.array(y)).to(torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.size(0)

print(y_train.unique())
data_train = TensorDataset(X_train[:10000],y_train[:10000])
data_test = TensorDataset(X_test[:1000],y_test[:1000])
train_loader = torch.utils.data.DataLoader(data_train, batch_size=50, shuffle=False)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=50, shuffle=False)