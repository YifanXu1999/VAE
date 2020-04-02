import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid
import math
latent_dim=20
h_dim = 20
num_layer_iaf = 10

trainloader = DataLoader(
    MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)
testloader = DataLoader(
    MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)

class AutoregressiveLinear(nn.Module):
    def __init__(self):
        super(AutoregressiveLinear, self).__init__()
        self.z_layer = nn.Linear(latent_dim, latent_dim)
        self.h_layer = nn.Linear(h_dim, latent_dim)
    def forward(self, z, h):
        return self.z_layer(z) + self.h_layer(h)

class IAFLinear(nn.Module):
    def __init__(self):
        super(IAFLinear,self).__init__()
        self.m_layer = AutoregressiveLinear()
        self.s_layer = AutoregressiveLinear()
    def forward(self, z, h):
        m = self.m_layer(z, h)
        s = self.s_layer(z, h)
        return m, s

class VAE_IAF(nn.Module):
    def __init__(self, hidden_dim=500):
        super(VAE_IAF, self).__init__()
        self.encoder_l1 = nn.Linear(784, hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.encoder_h = nn.Linear(hidden_dim, h_dim)

        self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 784)

        self.iaf_layers = nn.ModuleList(IAFLinear() for i in range(num_layer_iaf))

    def encode(self, x_in):
        x_in = self.encoder_l1(x_in)
        mean = self.encoder_mean (x_in)
        logvar = self.encoder_logvar(x_in)
        h = self.encoder_h(x_in)
        return mean, logvar, h

    def decoder(self, z):
        z = F.relu(self.decoder_l1(z))
        x_out = F.sigmoid(self.decoder_output(z))
    
        return x_out.view(-1, 1, 28, 28)
    def forward(self, x_in):
        mean, logvar, h = self.encode(x_in.view(-1, 784))
        eps = torch.rand_like(mean)
        z = torch.exp(logvar * 0.5) * eps + mean
        logqz_x = -torch.sum(logvar * 0.5 + 0.5 * eps * eps + 0.5 * math.log(math.pi * 2))
        for layer in self.iaf_layers:
            m, s = layer(z, h)
            std = torch.sigmoid(s)
            z = std * z + (1 - std) * m
            logqz_x = logqz_x - torch.sum(torch.log(std))
        x_out = self.decoder(z)
        logpz = -torch.sum(0.5*(z ** 2 + np.log(2 * math.pi)))
        logpx_z = -F.binary_cross_entropy(x_out, x_in, size_average=False)
        loss = -(logpx_z + logpz - logqz_x) / x_in.size(0)
        return x_out, loss

model = VAE_IAF()
print(model.parameters())
# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Training
def train(model,optimizer,dataloader,epochs=15):
    losses = []
    for epoch in range(epochs):
        print(epoch)
        for images,_ in dataloader:
            x_in = (images)
            
            optimizer.zero_grad()
            x_out, loss= model(x_in)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
    return losses

train_losses = train(model,optimizer,trainloader)
plt.figure(figsize=(10,5))
plt.plot(train_losses)
plt.show()

# Visualize VAE input and reconstruction
def visualize_mnist_vae(model,dataloader,num=16):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.axis('off')
        plt.show()
        
    images,_ = iter(dataloader).next()
    images = images[0:num,:,:]
    x_in = (images)
    x_out, _ = model(x_in)
    x_out = x_out.data
    imshow(make_image_grid(images))
    imshow(make_image_grid(x_out))

visualize_mnist_vae(model,testloader)