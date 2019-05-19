from __future__ import print_function, division
import os
import sys
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
                mode=self.mode)
        return y


def conv_output_dim(input_size,
        kernel_size, stride=1, padding=0, dilation=1, **kwargs):
    """Calculate the output dimension of a convolutional layer
    """
    from math import floor
    return floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


def conv_transposed_output_dim(input_size,
        kernel_size, stride=1, padding=0, dilation=1, **kwargs):
    """Calculate the output dimension of a transposed convolutional layer
    """
    return (input_size-1)*stride - 2*padding + dilation*(kernel_size-1) + 1


def convnet_to_dense_size(input_size, params_list):
    """Calculate the output size of a purely convolutional network
        assumes a square image input
    """
    size = input_size
    for params in params_list:
        size = conv_output_dim(size, **params)
    return size * size * params_list[-1]['out_channels']


def convnet_layers(input_channels, params_list):
    """Generate a list of fully convolutional layers
        given a list of parameters
    """
    layers = []
    for i in range(len(params_list)):
        in_channels = input_channels if i == 0 else params_list[i-1]['out_channels']
        layers += [
            nn.Conv2d(in_channels, **params_list[i]),
            nn.ReLU()
        ]
    return layers


def transposed_convnet_layers(input_channels, params_list):
    """Generate a list of fully transposed convolutional
        layers given a list of parameters
    """
    layers = []
    for i in range(len(params_list)):
        in_channels = input_channels if i == 0 else params_list[i-1]['out_channels']
        layers += [
            nn.ConvTranspose2d(in_channels, **params_list[i]),
            nn.ReLU() if (i < len(params_list) - 1) else nn.Sigmoid()
        ]
    return layers
