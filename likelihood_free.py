import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from bridge import BrownianBridge
from utils import SinusoidalPositionEmbeddings
#from denoising_diffusion_pytorch import Unet
import matplotlib.pyplot as plt
import numpy as np
from utils import dataSetLikelihoodFree
import wandb
import time
import scipy as sp


class Network(nn.Module):
    def __init__(self, dims_in, dims_out, dropout_rate =0.0):
        super().__init__()

        #drop_layer = nn.Dropout(p=dropout_rate)
        n_layers = len(dims_out)
        assert dims_out[:-1] == dims_in[1:], "Dimension of subsequent layer not matching"
        self.dim_in_out = zip(dims_in, dims_out)
        self.layers = nn.ModuleList([])
        for i, dims in enumerate(self.dim_in_out):
            dim_in, dim_out = dims
            self.layers.append(torch.nn.Linear(dim_in, dim_out))

            if i != n_layers-1:
                self.layers.append(torch.nn.LeakyReLU())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                print("NOT LAST LAYER ?")



    def forward(self, x):
        for h in self.layers:
            x = h(x)

        return x

def l2_loss(true_data, pred_data):
    return torch.mean((true_data - pred_data)**2)

def prior(N_sample):
    return np.random.uniform(-10, 10, size=(N_sample,1))

def likelihood(theta, N_sample):
    sigma = np.ones((N_sample,1))
    unifs = np.random.uniform(size=N_sample)
    sigma[unifs < 0.5] = np.sqrt(0.01)
    return theta + np.random.normal(size=(N_sample,1))*sigma


def sample_dataset(N_sample):
    theta = prior(N_sample)
    data = likelihood(theta, N_sample)
    return theta, data

def sample_pertubed_data(N_sample, brid):
    dataset_data, dataset_param = sample_dataset(N_sample)
    dataset_data = torch.tensor(dataset_data, dtype=torch.float32)
    dataset_param = torch.tensor(dataset_param, dtype=torch.float32)
    times = torch.rand((N_sample, 1))
    perturbed_dataset = brid.sample(times, torch.randn_like(dataset_param), dataset_param)
    return dataset_data , dataset_param,times, perturbed_dataset



def compute_prior(theta):
    if theta > - 10 and theta < 10:
        return 1/20

    return 0

def compute_likelihood(theta, observed_data):
    log_first_mode = -1/2*(observed_data - theta)**2 - 1/2 * np.log(2*np.pi) - 1/2 * np.log(1)
    log_second_mode = -1/2*(observed_data - theta)**2/0.01 - 1/2 * np.log(2*np.pi) - 1/2 * np.log(0.01)
    return 0.5*np.exp(log_first_mode) + 0.5*np.exp(log_second_mode)


def compute_partition_function(observed_data):
    first_mode = 1/40 * (sp.stats.norm.cdf(10, loc=observed_data, scale=1) - sp.stats.norm.cdf(-10, loc=observed_data, scale=1))
    second_mode = 1/40 * (sp.stats.norm.cdf(10, loc=observed_data, scale=np.sqrt(0.01)) - sp.stats.norm.cdf(-10, loc=observed_data, scale=np.sqrt(0.01)))
    return first_mode + second_mode


def run(retrain=True, data_path="data/likelihood_free/"):
    brid = BrownianBridge(1, a=25, b=5)
    if retrain:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        # brid = BrownianBridge(1, a=1, b=3)

        dataset_data, dataset_param, times, perturbed_dataset = sample_pertubed_data(500000, brid)
        torch.save(dataset_data, data_path + "dataset_data")
        torch.save(dataset_param, data_path + "dataset_param")
        torch.save(times, data_path + "times")
        torch.save(perturbed_dataset, data_path + "perturbed_dataset")

        dataset_data_test, dataset_param_test, times_test, perturbed_dataset_test = sample_pertubed_data(5000, brid)
        torch.save(dataset_data_test, data_path + "dataset_data_test")
        torch.save(dataset_param_test, data_path + "dataset_param_test")
        torch.save(times_test, data_path + "times_test")
        torch.save(perturbed_dataset_test, data_path + "perturbed_dataset_test")

        batch_size = 50
        dataset = dataSetLikelihoodFree(dataset_data, dataset_param, perturbed_dataset, times)
        dataLoad = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net = Network([3, 256, 256, 256], [256, 256, 256, 1])
        net = net.to(device)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.00003)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.00003)
        epochs = 10000
        input_test = torch.concat([dataset_data_test, perturbed_dataset_test, times_test], dim=-1)
        input_test = input_test.to(device)
        dataset_param_test = dataset_param_test.to(device)

        for n_epoch in range(epochs):
            data = iter(dataLoad)
            all_losses = []
            start = time.time()
            for n_batch in range(10000):
                data_batch, param_batch, perturbed_param_batch, time_batch = next(data)
                data_batch = data_batch.to(device)
                batch_size = data_batch.shape[0]
                input_batch = torch.concat([data_batch, perturbed_param_batch, time_batch], dim = -1)
                input_batch = input_batch.to(device)
                pred_data = net.forward(input_batch)
                loss = l2_loss(param_batch, pred_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.detach().cpu().numpy())

            end = time.time()
            print("Time on epoch:", end - start)
            print("Train Loss:", np.mean(all_losses))
            net.eval()
            pred_test = net.forward(input_test)
            loss_test = l2_loss(dataset_param_test, pred_test)
            print("EPOCH:", n_epoch)
            print("LOSS TEST", torch.sqrt(loss_test))
            torch.save(net, data_path + "network")
            #wandb.log({"train_losses": all_losses, "test_loss": loss_test.detach().cpu().numpy()})
            net.train()
            print("\n\n\n")


    unet = torch.load(data_path + "network", map_location=torch.device('cpu'))
    unet.eval()
    times = torch.tensor(np.linspace(0, 1, 1000), dtype=torch.float32)[:, None]
    traj, test = brid.euler_maruyama(torch.randn(100000, 1),times[:, :, None], 1, unet, observation=torch.ones((100000,1), dtype=torch.float32)*8)

    #np.save("data/gaussian/generatedData2.npy", test[:, 0].detach().numpy())
    print("\n\n")
    print(np.mean(test[:, 0].detach().numpy()))
    print(np.var(test[:, 0].detach().numpy()))
    plt.hist(test[:, 0].detach().numpy(), density=True, alpha=0.5, bins= 60)
    xx = np.linspace(3, 13, 10000)
    observed_data = 8
    y = np.array([compute_prior(x)*compute_likelihood(x, observed_data)/compute_partition_function(observed_data) for x in xx])
    #y = np.array([compute_likelihood(x, observed_data) for x in xx])
    #plt.hist(sampled_posterior[:, -1, 0], bins=40, density=True, alpha=0.5)
    #plt.hist(sorted_params[:10000, 0], bins=40, density=True, alpha=0.5)
    plt.plot(xx, y)
    plt.show()


if __name__=="__main__":
    #d1 = np.load("data/gaussian/generatedData1.npy")
    #d2 = np.load("data/gaussian/generatedData2.npy")
    #plt.boxplot([d1, d2], showfliers=False)
    #plt.show()

    run(retrain=False)