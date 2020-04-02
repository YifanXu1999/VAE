import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

# Data loaders
trainloader = DataLoader(
    MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)


class VAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=500):
        super(VAE, self).__init__()
        self.encoder_l1 = nn.Linear(784, hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 784)

    def encode(self, x_in):
        x = F.relu(self.encoder_l1(x_in.view(-1, 784)))
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar
    
    def decode(self, z):
        z = F.relu(self.decoder_l1(z))
        x_out = F.sigmoid(self.decoder_output(z))
        return x_out.view(-1, 1, 28, 28)
    
    def sample(self, mu, log_var):
        # z = mu + standard deviavation * eps
        eps = torch.normal(torch.zeros(size=mu.size()), torch.ones(size=log_var.size()))
        sd = torch.exp(log_var * 0.5)
        z = mu + sd * eps
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encode(x_in)
        z = self.sample(z_mean, z_logvar)
        x_out = self.decode(z)
        return x_out, z_mean, z_logvar

# Loss function
def criterion(x_out, x_in, z_mu, z_logvar):
    # ELBO = -DK(q(z|x)|| p(z)) + logp_theta(x|z)
    #      = 1/2(1 + log(var) - mu ^2 - var) +  logp_theta(x|z)
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size
    return loss

model = VAE()
# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Training
def train(model,optimizer,dataloader,epochs=15):
    losses = []
    for epoch in range(epochs):
        for images,_ in dataloader:
            x_in = (images)
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion(x_out,x_in,z_mu,z_logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
    print('Vanilla VAE mean loss', np.asarray(losses).mean())
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
    x_out,_,_ = model(x_in)
    x_out = x_out.data
    imshow(make_grid(images))
    imshow(make_grid(x_out))

visualize_mnist_vae(model,testloader)