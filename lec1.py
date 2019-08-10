import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.datasets import load_iris, load_digits

t = torch.tensor([[1, 2], [3, 4.]], device='cuda:0')

t.cpu().numpy()

## Linear Regression
w_true = torch.Tensor([1, 2, 3])

X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1) # bias
y = torch.mv(X, w_true) + torch.randn(100) * 0.5
w = torch.randn(3, requires_grad=True)

gamma = 0.1

losses = []

for epoch in range(100):
    w.grad = None
    y_pred = torch.mv(X, w)
    loss = torch.mean((y - y_pred) ** 2)
    loss.backward()
    
    w.data = w.data - gamma * w.grad.data
    
    losses.append(loss.item())
    
    
plt.plot(losses)
plt.show()

## Linear Regression nn, optim
net = nn.Linear(in_features=3, out_features=1, bias=False)
optimizer = optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
losses = []
for i in range(100):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
plt.plot(losses)
plt.show()

## logistic regression
iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

net = nn.Linear(4, 1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.25)
losses = []

for i in range(100):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
plt.plot(losses)
plt.show()

mnist = load_digits()
X = mnist.data
Y = mnist.target

X = torch.tensor(X, device="cuda:0", dtype=torch.float32)
Y = torch.tensor(Y, device="cuda:0", dtype=torch.int64)

net = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10))
net.to("cuda:0")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
losses = []

for i in range(300):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
plt.plot(losses)
plt.show()
















