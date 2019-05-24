import torch
from torch import nn
from torch.nn import functional as F

class VAE_Simple(nn.Module):

    def __init__(self, z_dim=20, img_size=28):
        super(VAE_Simple, self).__init__()
        # encoder
        self.input_size = img_size**2
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, z_dim) # mean
        self.fc22 = nn.Linear(400, z_dim) # std
        # decoder
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
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
        x_flat = x.view(-1, x.shape[2]*x.shape[3])
        BCE = F.binary_cross_entropy(recon_x, x_flat, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    @property
    def total_parameters(self):
        return sum([torch.numel(p) for p in self.parameters()])

# print(VAE_Simple().total_parameters)
