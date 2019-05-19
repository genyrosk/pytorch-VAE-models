
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from data import DisentangledSpritesDataset

from models import VAE_SuperResolution

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


v = VAE_SuperResolution(z_dim=20, img_channels=1, img_size=64)
input = torch.zeros(128,1, 64,64)

d = nn.Sequential(
    nn.Conv2d(1, 8, (3,3), stride=(1,1), padding=1),
    nn.ELU()
)
x = d(input)
print(x.shape)
# nn.Conv2d(32, 32, (3,3), stride=(1,1), padding=1),
# nn.ELU(),
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
    nn.Conv2d(32, 64, (5,5), stride=(2,2), padding=2),
    nn.ELU()
)
x = d(x)
print(x.shape)
d = nn.Sequential(
    nn.Conv2d(64, 128, (5,5), stride=(1,1), padding=2),
    nn.ELU()
)
x = d(x)
print(x.shape)
d = Flatten()
x = d(x)
print(x.shape)

# s = v.encode(input)[0].shape
# print(s)


n_channels = 128
d = UnFlatten(n_channels)
x = d(x)
print(x.shape)
# nn.Conv2d(n_channels, n_channels, (5,5), (1,1), padding=2),
# nn.ELU(),

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(int(n_channels/4), 32, (5,5), (1,1), padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(32, 32, (5,5), (1,1), padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)
# nn.Conv2d(8, 8, (5,5), (1,1), padding=2),
# nn.ELU(),
d = nn.Sequential(
nn.Conv2d(8, 8, (5,5), (1,1), padding=2),
nn.ELU())
x = d(x)
print(x.shape)

d = nn.PixelShuffle(upscale_factor=2)
x = d(x)
print(x.shape)

d = nn.Sequential(
nn.Conv2d(2, 1, (5,5), (1,1), padding=2),
nn.Sigmoid())
x = d(x)
print(x.shape)
