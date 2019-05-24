from __future__ import print_function, division
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

dirName = 'results_super_resolution_zdim-6_beta-5'
parent_dir = os.path.join(os.path.dirname(__file__), '..')
model_name = 'super_resolution_model'
model = load_checkpoint(f'{dirName}/{model_name}.pth')

dims = 6
nums = 11
x_range = np.linspace(-3,3,nums)
z = np.zeros((dims, nums, dims), dtype=np.float32)

for dim in range(dims):
    z[dim, :, dim] = x_range

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z = torch.tensor(z).view(dims*nums, z.shape[-1]).to(device)
sample = model.decode(z).cpu()
save_image(sample.view(dims*nums, 1, 64, 64),
           f'{dirName}/sample_latent_space.png',
           nrow=nums)
