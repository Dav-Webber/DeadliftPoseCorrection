import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MyDataset(Dataset):
    def __init__(self, root):
        self.df = pd.read_csv(root)

    def __getitem__(self, idx):
        y = self.df['class']
        x = self.df.drop(columns=['class'])
        # normalize
        xval = x.values
        min_max_scaler = preprocessing.MinMaxScaler()
        xval_scaled = min_max_scaler.fit_transform(xval)
        # convert to torch tensors
        xval_scaled = pd.DataFrame(xval_scaled)
        tensorX = torch.tensor(xval_scaled.to_numpy()).float()
        tensorY = torch.tensor(y.to_numpy()).float()
        return tensorX[idx], tensorY[idx]

    def __len__(self):
        return self.df.shape[0]


trainData = MyDataset('/content/drive/MyDrive/csvCdata/train-c3.csv')
testData = MyDataset('/content/drive/MyDrive/csvCdata/test-c3.csv')

train_loader = DataLoader(dataset=trainData, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=testData, batch_size=1, shuffle=True)
data_iter = iter(train_loader)
data = data_iter.next()
features, labels = data

num_epochs = 1
total_samples = len(trainData)
n_iterations = math.ceil(total_samples / 1)
print(total_samples, n_iterations)


# for epoch in range(num_epochs):
#   for i , (inputs,labels) in enumerate(data_loader):
#   #forward then backward pass


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


net = Net(12)
#
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(num_epochs):
    train_correct = 0
    print(f'EPOCH : {epoch + 1}')
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = Variable(inputs), Variable(labels).long()
        net.train()
        optimizer.zero_grad()
        y_pred = net(inputs)

        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        max_value = torch.max(y_pred).item()
        n = 0
        for j in range(len(y_pred[0])):
            if (y_pred[0][j].item() == max_value):
                n = j
        if n == labels.item():
            train_correct += 1
        accuracy = train_correct / (i + 1)
        print(
            f'TRAIN -- data row: {i + 1}, prediction:{n}, truth:{labels.item()}, accuracy:{accuracy * 100}%, loss:{round_tensor(loss)}')

    test_correct = 0

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = Variable(inputs), Variable(labels).long()

            net.eval()
            outputs = net(inputs)
            max_value = torch.max(outputs).item()
            n = 0
            for j in range(len(outputs[0])):
                if (outputs[0][j].item() == max_value):
                    n = j
            if n == labels.item():
                test_correct += 1
            test_accuracy = test_correct / (i + 1)
        print(f'TEST -- accuracy:{test_accuracy * 100}%')
