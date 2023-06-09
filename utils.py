import torch
from torch import nn
import math
from inspect import isfunction
from torch.utils.data import Dataset



def l2_loss_image(true_images, images):
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



class dataSetLikelihoodFree(Dataset):

    def __init__(self, data, param, perturbed_param, times):
        self.data = data
        self.param = param
        self.perturbed_param = perturbed_param
        self.times = times

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.param[idx], self.perturbed_param[idx], self.times[idx]


class dataSetMultiple(Dataset):

    def __init__(self, images, perturbed_images, times, dataset_num):
        self.images = images
        self.perturbed_images = perturbed_images
        self.times = times
        self.dataset_num = dataset_num

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.perturbed_images[idx], self.times[idx], self.dataset_num[idx]


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



def sample_pertubed_data(data, brid, number_dataset):
    all_times = []
    all_perturbed_dataset = []
    for _ in range(number_dataset):
        times = torch.rand((data.shape[0], 1))
        perturbed_dataset = brid.sample(times, torch.zeros_like(data), data)
        all_times.append(times)
        all_perturbed_dataset.append(perturbed_dataset)

    all_times = torch.concat(all_times, dim=0)
    all_perturbed_dataset = torch.concat(all_perturbed_dataset, dim=0)
    return all_perturbed_dataset, all_times


def l2_loss(true_data, pred_data):
    return torch.mean(torch.sum((true_data - pred_data)**2, dim=-1))
