import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mnist_models import VAE_SuperResolution

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


# v = VAE_SuperResolution(z_dim=20, img_channels=1, img_size=28)
x = torch.zeros(128,1, 64,64)
print(x.shape)

d = nn.Sequential(
    nn.Conv2d(1, 8, (3,3), stride=(2,2), padding=1),
    nn.ELU()
)
x = d(x)
print(x.shape)

d = nn.Sequential(
    nn.Conv2d(8, 16, (4,4), stride=(2,2), padding=1),
    nn.ELU()
)
x = d(x)
print(x.shape)

d = nn.Sequential(
    nn.Conv2d(16, 32, (5,5), stride=(2,2), padding=2),
    nn.ELU()
)
x = d(x)
print(x.shape)

d = nn.Sequential(
    nn.Conv2d(32, 64, (5,5), stride=(1,1), padding=2),
    nn.ELU()
)
x = d(x)
print(x.shape)

d = Flatten()
x = d(x)
print(x.shape)

# s = v.encode(input)[0].shape
# print(s)

d = UnFlatten(64)
x = d(x)
print(x.shape)


d = nn.Sequential(
nn.Conv2d(64, 64, (5,5), stride=1, padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(16, 64, (5,5), stride=1, padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(16, 64, (5,5), stride=1, padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(16, 64, (5,5), stride=1, padding=2),
nn.ReLU())
x = d(x)
print(x.shape)
