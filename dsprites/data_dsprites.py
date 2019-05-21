from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, []


def load_dsprites(dir='/home/genyrosk/datasets',
                val_split=0.9, shuffle=True, seed=42, batch_size=64):
    # img_size = 64
    path = os.path.join(dir, 'dsprites-dataset')
    dataset = DisentangledSpritesDataset(path, transform=transforms.ToTensor())

    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader
