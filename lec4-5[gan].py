import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from torch import nn, optim
from statistics import mean
from tqdm import tqdm

img_data = ImageFolder("d:/dataset/oxford-102", transform=transforms.Compose([transforms.Resize(80),
                                                                              transforms.CenterCrop(64),
                                                                              transforms.ToTensor()]))

batch_size = 64
img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)

nz = 100
ngf = 32
ndf = 32

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=ngf*8),
                nn.ReLU(inplace=True),
                                  
                nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=ngf*4),
                nn.ReLU(inplace=True),
                  
                nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=ngf*2),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=ngf),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(in_channels=ngf, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh())
    
    def forward(self, x):
        out = self.main(x)
        return out
 

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                
                nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                
                nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                
                nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, x):
        out = self.main(x)
        return out.squeeze()
    
if __name__ == "__main__":
    device = 'cuda:0'
    d = DNet().to(device)
    g = GNet().to(device)
    
    opt_d = optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_g = optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)
    loss_f = nn.BCEWithLogitsLoss()
    
    fixed_z = torch.randn(batch_size, nz, 1, 1).to(device)
    
def train_dcgan(g, d, opt_g, opt_d, loader):
    # log tracking
    log_loss_g = []
    log_loss_d = []
    for real_img, _ in tqdm(loader):
        batch_len = len(real_img)
        # real_img
        real_img = real_img.to(device)
        
        # fake_img
        z = torch.randn(batch_len, nz, 1, 1).to(device)
        fake_img = g(z)
        fake_img_tensor = fake_img.detach()
        out = d(fake_img)
        
        loss_g = loss_f(out, ones[: batch_len])
        log_loss_g.append(loss_g.item())
        
        # both d,g update sync --> zero_grad
        d.zero_grad(), g.zero_grad()
        loss_g.backward()
        opt_g.step()
        
        real_out = d(real_img)
        loss_d_real = loss_f(real_out, ones[:batch_len]) # cross entropy 계산 , dummy 행렬
        
        fake_img = fake_img_tensor
        fake_out = d(fake_img_tensor)
        loss_d_fake = loss_f(fake_out, zeros[:batch_len])
        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())
        
        d.zero_grad(), g.zero_grad()
        loss_d.backward()
        opt_d.step()
    
    return mean(log_loss_g), mean(log_loss_d)

for epoch in range(300):
    train_dcgan(g, d, opt_g, opt_d, img_loader)
    if epoch % 10 == 0:
        torch.save(obj=g.state_dict(), f="d:/data/g_{:03d}.prm".format(epoch), pickle_protocol=4)
        torch.save(obj=d.state_dict(), f="d:/data/d_{:03d}.prm".format(epoch), pickle_protocol=4)
        generated_img = g(fixed_z)
        save_image(generated_img, "d:/data/{:03d}.jpg".format(epoch))















                 
                                  
        