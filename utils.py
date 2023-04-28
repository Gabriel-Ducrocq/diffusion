import torch
from torch import nn
import math
from inspect import isfunction
from torch.utils.data import Dataset



def l2_loss(true_images, images):
    return torch.mean(torch.sum((true_images - images)**2, dim=(-1,-2)))


class dataSet(Dataset):

    def __init__(self, images, perturbed_images, times):
        self.images = images
        self.perturbed_images = perturbed_images
        self.times = times

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.perturbed_images[idx], self.times[idx]


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Get the positional encoding of times t
        :param time: torch.tensor (N_batch,) of float, corresponding to the sampling times
        :return: torch.tensor (N_batch, dim), the positionql encodings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings