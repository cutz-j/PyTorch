from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim
import torch
import tqdm
import math
from torchvision.utils import save_image

class DownSizedPariImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)
        
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)
        
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)
        return small_img, large_img
    
train_data = DownSizedPariImageFolder("d:/dataset/lfw-deepfunneled/train",
                                      transform=transforms.ToTensor())
test_data = DownSizedPariImageFolder("d:/dataset/lfw-deepfunneled/test",
                                     transform=transforms.ToTensor())

batch_size = 32
device = 'cuda:0'
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

net = nn.Sequential(
        nn.Conv2d(3, 256, 4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 512, 4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU())

def psnr(mse, max_v=1.0):
    return 10 * math.log10(max_v**2 / mse)

def eval_net(net, data_loader, device='cuda:0'):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = net(x)
        ys.append(y)
        ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    score = nn.functional.mse_loss(ypreds, ys).item()
    return score

def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam, loss_fn=nn.MSELoss(),
              n_iter=10, device='cuda:0'):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        score = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
        train_losses.append(running_loss / len(train_loader))
        val_acc.append(eval_net(net, test_loader, device))
        print(epoch, train_losses[-1], psnr(train_losses[-1]), psnr(val_acc[-1]), flush=True)
        
net.to(device)
train_net(net, train_loader, test_loader, device=device)

random_test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
it = iter(random_test_loader)
x, y = next(it)

bl_recon = torch.nn.functional.upsample(x, 128, mode="bilinear", align_corners=True)
yp = net(x.to(device).to("cpu"))
save_image(torch.cat([y, bl_recon, yp], 0), "cnn_upscale.jpg", nrow=4)
























