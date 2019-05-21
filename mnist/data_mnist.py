from __future__ import print_function
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist(shuffle=True, batch_size=64):
    # img_size = 28
    train_loader = DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.ToTensor()
            ), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
            datasets.MNIST('../data', train=False,
                transform=transforms.ToTensor()
            ), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
