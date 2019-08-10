from torch import nn, optim, tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch

device = 'cuda:0'
mnist = load_digits()
X = mnist.data
Y = mnist.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

X_train = tensor(X_train, device=device, dtype=torch.float32)
Y_train = tensor(Y_train, device=device, dtype=torch.int64)
X_test = tensor(X_test, device=device, dtype=torch.float32)
Y_test = tensor(Y_test, device=device, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()
ds = TensorDataset(X_train, Y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)
k = 100
net = nn.Sequential(
        nn.Linear(64, k),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(k, k),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(k, k),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(k, k),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(k, 10))

net.to(device)
optimizer = optim.Adam(net.parameters())
train_losses, test_losses = [], []
for epoch in range(100):
    running_loss = 0.0
    net.train()
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    net.eval()
    y_pred = net(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())
        
        
        
## custom layer
class Custom(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

mlp = nn.Sequential(
        Custom(64, 200),
        Custom(200, 200),
        Custom(200, 200),
        nn.Linear(200, 10))


        
        
        
        
        
        
        
        
        
        
    