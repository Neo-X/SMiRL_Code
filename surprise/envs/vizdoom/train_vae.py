from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from networks import VAE
import pdb

class BufferDataset(Dataset):
    def __init__(self, train=True):
        self.datapath = './data'
        self.train = train

    def __getitem__(self, idx):
        if not self.train:
            idx += 45000
        data = np.load('./data/buffer/{:07d}.npy'.format(idx))
        return data

    def __len__(self):
        if self.train:
            return 45000
        else:
            return 4900

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 130), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + .1 * KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(0)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(0)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(8, 1, 10, 13),
                                      recon_batch.view(8, 1, 10, 13)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def save_checkpoint(state, filename='checkpoints/vae.pth'):
    torch.save(state, filename)

batch_size = 64
epochs = 10000

train_loader = DataLoader(BufferDataset(BufferDataset()), batch_size=batch_size)
test_loader = DataLoader(BufferDataset(BufferDataset(train=False)), batch_size=8)

model = VAE().to(0)
#chkpt = torch.load('checkpoints/vae.pth')['state_dict']
#model.load_state_dict(chkpt)
optimizer = optim.Adam(model.parameters(), lr=1e-4)



for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(0)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 10, 13),
                   'results/sample_' + str(epoch) + '.png')
    save_checkpoint({'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()})
