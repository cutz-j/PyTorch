from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim
import torch
import tqdm

## param ##
batch_size = 32
device = 'cuda:0'

train_imgs = ImageFolder("d:/dataset/taco_and_burrito/train/",
                         transform=transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()]))
test_imgs = ImageFolder("d:/dataset/taco_and_burrito/test/",
                        transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
train_loader = DataLoader(train_imgs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_imgs, batch_size=32, shuffle=False)

print(train_imgs.classes)

net = models.resnet18(pretrained=True)
for p in net.parameters():
    p.requires_grad = False

fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, 2)

def eval_net(net, data_loader, device='cuda:0'):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
    
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam, only_fc=True, loss_fn=nn.CrossEntropyLoss(),
              n_iter=10, device='cuda:0'):
    train_losses = []
    train_acc = []
    val_acc = []
    if only_fc:
        optimizer = optimizer_cls(net.fc.parameters())
    else:
        optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        train_acc.append(n_acc / n)
        
        val_acc.append(eval_net(net, test_loader, device))
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)
        
        
net.to(device)
train_net(net, train_loader, test_loader, n_iter=20, device=device)

class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

net = models.resnet18(pretrained=True)
for p in net.parameters():
    p.requires_grad = False
net.fc = IdentityLayer()

conv_net = nn.Sequential(
        nn.Conv2d(3, 32, 5),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 5),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        FlattenLayer())

test_input = torch.ones(1, 3, 224, 224)
conv_output_size = conv_net(test_input).size()[-1]

net = nn.Sequential(conv_net, nn.Linear(conv_output_size, 2))
net.to(device)
train_net(net, train_loader, test_loader, n_iter=10, only_fc=False, device=device)












