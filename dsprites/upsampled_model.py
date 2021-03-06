import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    def forward(self, input):
        size = int((input.size(1) // self.n_channels)**0.5)
        return input.view(input.size(0), self.n_channels, size, size)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        y = self.interp(x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=False)
        return y

class VAE_Upsampled(nn.Module):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, z_dim=20, img_channels=1, img_size=64):
        super(VAE_Upsampled, self).__init__()

        ## encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 8, (3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5,5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )

        ## output size depends on input image size
        demo_input = torch.ones([1,img_channels,img_size,img_size])
        h_dim = self.encoder(demo_input).shape[1]
        print('h_dim', h_dim)
        ## map to latent z
        # h_dim = convnet_to_dense_size(img_size, encoder_params)
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)

        ## decoder
        self.fc2 = nn.Linear(z_dim, h_dim)
        n_channels = 64
        self.decoder = nn.Sequential(
            UnFlatten(n_channels),
            Interpolate(scale_factor=(2,2), mode='bilinear'),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            Interpolate(scale_factor=(2,2), mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            Interpolate(scale_factor=(2,2), mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            Interpolate(scale_factor=(2,2), mode='bilinear'),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        img = self.decoder(self.fc2(z))
        return img

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=5.0):
        """Reconstruction + KL divergence losses summed over all elements (of a batch)
            see Appendix B from VAE paper:
            Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            https://arxiv.org/abs/1312.6114
            KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    @property
    def total_parameters(self):
        return sum([torch.numel(p) for p in self.parameters()])

# print(VAE_Upsampled().total_parameters)
