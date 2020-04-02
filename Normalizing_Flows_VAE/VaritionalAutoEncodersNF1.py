'''
vanilla
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid

latent_dim=40
numOfTransformation=30

trainloader = DataLoader(
    MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)
testloader = DataLoader(
    MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)
class Flow(nn.Module):

    def __init__(self):
        super(Flow, self).__init__()

    def forward(self,z, w, u, b):
        '''
        f(z) = z + u * tanh(w^t * z + b)
        f'(z) = 1 + u(1 - tanh(w^t * z + b)  ^ 2) w
        '''
        # transf = tanh(w^t * z + b)
        transf = F.tanh(
            (w * z).sum(dim=1, keepdim=True) + b
        )
        f_z = z + u * transf

        # Inverse
        # psi_z = tanh' (w^T z + b) w
        psi_z = (1 - transf ** 2) * w
        # log_abs_det_jacobian = |1 + u.dot(psi_z)|
        log_abs_det_jacobian = torch.log(
            (1 + (psi_z * u).sum(dim=1, keepdim=True)).abs()
        )
        return f_z, log_abs_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.flows = nn.ModuleList([Flow() for i in range(numOfTransformation)])
    def forward(self, z_k, w, u, b):
        # ladj -> log abs det jacobian
        sum_ladj = 0
        for i, flow in enumerate(self.flows):
            z_k, ladj_k = flow(z_k, w[:, i * latent_dim : (i + 1) * latent_dim], u[:, i * latent_dim : (i + 1) * latent_dim], b[:, i: i+1])
            sum_ladj += ladj_k

        return z_k, sum_ladj

class VAE_NF(nn.Module):

    def __init__(self, hidden_dim=500):
        super(VAE_NF, self).__init__()
        self.encoder_l1 = nn.Linear(784,hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 784)
        self.nf_w = nn.Linear(hidden_dim, latent_dim * numOfTransformation)
        self.nf_u = nn.Linear(hidden_dim, latent_dim * numOfTransformation)
        self.nf_b = nn.Linear(hidden_dim, numOfTransformation)
        self.latent_dim = latent_dim
        self.nf = NormalizingFlow()
    def encode(self, x_in):
        x_in = self.encoder_l1(x_in.view(-1, 784))
        mean = self.encoder_mean(x_in)
        logvar = self.encoder_logvar(x_in)
        return mean , logvar
    
    def getw_u_b(self, x_in):
        x_in = self.encoder_l1(x_in.view(-1, 784))
        w = self.nf_w(x_in)
        u = self.nf_u(x_in)
        b = self.nf_b(x_in)
        return w, u, b
    
    def decode(self, z):
        z = F.relu(self.decoder_l1(z))
        x_out = F.sigmoid(self.decoder_output(z))
        return x_out.view(-1, 1, 28, 28)
    
    def sample(self, mu, logvar):
        eps = torch.rand_like(logvar)
        sd = torch.exp(logvar * 0.5)
        z = mu + sd * eps
        return z, eps
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z, eps = self.sample(mean, logvar)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        w, u, b = self.getw_u_b(x)
        z, sum_ladj = self.nf(z, w, u, b)
        kl_div = kl_div / x.size(0) - sum_ladj.mean()
        x_out = self.decode(z)
        return x_out, kl_div


def criterion(x_out, x_in, kld_loss):
    # ELBO = -DK(q(z|x)|| p(z)) + logp_theta(x|z)
    #      = 1/2(1 + log(var) - mu ^2 - var) +  logp_theta(x|z)
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False) / x_out.size(0)
    loss = (bce_loss + kld_loss) # normalize by batch size
    return loss

model = VAE_NF()
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
            x_out, kld_loss= model(x_in)
            loss = criterion(x_out,x_in,kld_loss)
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
    x_out,_ = model(x_in)
    x_out = x_out.data
    imshow(make_image_grid(images))
    imshow(make_image_grid(x_out))

visualize_mnist_vae(model,testloader)