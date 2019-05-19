from __future__ import print_function, division
import os
import sys
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models import VAE_Simple, VAE_Conv, VAE_Upsampled, VAE_SuperResolution
from data import DisentangledSpritesDataset

parser = argparse.ArgumentParser(description='beta-VAE MNIST / dSprites')
parser.add_argument('model_name', type=str, default='simple', metavar='MODEL',
                    help='model name (default: simple)', nargs='?')
parser.add_argument('--data', type=str, default='MNIST', metavar='D',
                    help='dataset name (default: MNIST, also: dSprites)')
parser.add_argument('--z-dim', type=int, default=15, metavar='Z',
                    help='number of latent variables z (default: 15)')
parser.add_argument('--beta', type=int, default=5.0, metavar='B',
                    help='regularisation coefficient * the KLD (default: 5.0)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

# Parse args
args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'Running on GPU: {args.cuda}')
print('Arguments:')
for arg, val in args._get_kwargs():
    print(f'    {arg:14s} {val}')

# select model
VAE_models = {
    'simple': VAE_Simple,
    'conv': VAE_Conv,
    'upsampled': VAE_Upsampled,
    'super_resolution': VAE_SuperResolution
}
VAE_model = VAE_models[args.model_name]
dirName = f'results_{args.data}_{args.model_name}_zdim-{args.z_dim}_beta-{args.beta}'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print(f'Directory {dirName} created \n')
else:
    print(f'Directory {dirName} already exists \n')

# Load MNIST dataset
if args.data == 'MNIST':
    img_size = 28
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.ToTensor()
            ), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                transform=transforms.ToTensor()
            ), batch_size=args.batch_size, shuffle=True)
# Load dSprites dataset
elif args.data == 'dSprites':
    img_size = 64
    dataset = DisentangledSpritesDataset('/home/genyrosk/datasets/dsprites-dataset')
    validation_split = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size, sampler=test_sampler)
else:
    raise Exception('Dataset not found. Try: MNIST, dSprites')

# model + optimizer + learning rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE_model(z_dim=args.z_dim, img_size=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ReduceLROnPlateau, StepLR
# scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                             mode='min', factor=0.1, patience=2,
#                             verbose=True)
loss_function = VAE_model.loss_function

print(f'Total parameters: {model.total_parameters}\n')

# plots
train_losses = []
test_losses = []
# fig, ax = plt.subplots(1,1,figsize=(12,8))

def train(epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # forward pass
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss + grads backprop
        loss = loss_function(recon_batch, data, mu, logvar, beta=args.beta)
        loss.backward()
        # save
        running_loss += loss.item()
        train_losses.append(loss.item())
        # update weights
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    # update learning rate
    scheduler.step()
    # print
    avg_loss = running_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta=args.beta)
            test_loss += loss.item()
            test_losses.append(loss.item())
            if i == 0:
                n = min(data.size(0), 8)
                recon_batch = recon_batch.view(args.batch_size, 1, img_size, img_size)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(),
                         f'{dirName}/reconstruction_{str(epoch)}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    for param_group in optimizer.param_groups:
        print(f'====> Learning rate: {param_group["lr"]:.7f}')

    # samples
    with torch.no_grad():
        sample = torch.randn(64, args.z_dim).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, img_size, img_size),
                   f'{dirName}/sample_{str(epoch)}.png')

# save model
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}
model_out_path = f"{dirName}/{args.model_name}_model.pth"
torch.save(checkpoint, model_out_path)
print(f"Model saved to {model_out_path}")
