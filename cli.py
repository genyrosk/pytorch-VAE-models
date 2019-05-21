import argparse

parser = argparse.ArgumentParser(description='beta-VAE MNIST / dSprites')
parser.add_argument(
    'model_name',
    type=str,
    default='simple',
    metavar='MODEL',
    nargs='?',
    help='model name (default: simple)'
)
parser.add_argument(
    '--data',
    type=str,
    default='MNIST',
    metavar='D',
    help='dataset name (default: MNIST, also: dSprites)'
)
parser.add_argument(
    '--z-dim',
    type=int,
    default=15,
    metavar='Z',
    help='number of latent variables z (default: 15)'
)
parser.add_argument(
    '--beta',
    type=int,
    default=5.0,
    metavar='B',
    help='regularisation coefficient * the KLD (default: 5.0)'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    metavar='N',
    help='input batch size for training (default: 128)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='enables CUDA training'
)
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)'
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    metavar='N',
    help='how many batches to wait before logging training status'
)
