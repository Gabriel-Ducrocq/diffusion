import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from bridge import BrownianBridgeArbitrary
from utils import SinusoidalPositionEmbeddings
# from denoising_diffusion_pytorch import Unet
import matplotlib.pyplot as plt
import numpy as np
from utils import dataSet
import wandb

#wandb.init(
#    project="diffusion-gaussian-example",

#    config={
#        "learning_rate": 0.003,
#        "optimizer": "adam",
#        "architecure": "MLP",
#        "size": [256, 256, 256, 1],
#        "epochs": 10
#    }
#)

def generate_dataset(N_sample):
    #start_times = torch.floor(torch.rand(size=(N_sample, 1)) * 2)
    start_times = torch.zeros(size=(N_sample, 1)) + 1
    end_times = start_times + 1
    dataset_zero = torch.zeros(size=(N_sample, 1))
    dataset_one = torch.tensor(np.random.normal(size=(N_sample, 1)), dtype=torch.float32) + 1
    dataset_two = torch.tensor(np.random.normal(size=(N_sample, 1)), dtype=torch.float32) + 1
    dataset_start = dataset_zero
    #dataset_start[start_times == 1.0] = dataset_one[start_times == 1.0]
    dataset_end = dataset_one
    dataset_end[end_times == 2.0] = dataset_two[end_times == 2.0]

    times = torch.rand((N_sample, 1)) + start_times
    """
    print(dataset_start)
    print(torch.unique(end_times[start_times==0]))
    print(torch.unique(end_times[start_times==1]))
    plt.boxplot(times[start_times ==0])
    plt.boxplot(times[start_times == 1])
    plt.title("Distrib  start times")
    plt.show()

    plt.hist(dataset_start[start_times==0], density=True, alpha=0.5, bins=500)
    plt.hist(dataset_start[start_times==1], density=True, alpha=0.5, bins=500)
    plt.title("Distrib  start dataset")
    plt.show()

    plt.hist(dataset_end[start_times==0], density=True, alpha=0.5, bins=500)
    plt.hist(dataset_end[start_times==1], density=True, alpha=0.5, bins=500)
    plt.title("Distrib  end dataset")
    plt.show()

    plt.hist(dataset_end[end_times==1], density=True, alpha=0.5, bins=500)
    plt.hist(dataset_end[end_times==2], density=True, alpha=0.5, bins=500)
    plt.title("Distrib  end dataset again")
    plt.show()
    """

    return start_times, end_times, times, dataset_start, dataset_end


class Network(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()
        n_layers = len(dims_out)
        assert dims_out[:-1] == dims_in[1:], "Dimension of subsequent layer not matching"
        self.dim_in_out = zip(dims_in, dims_out)
        self.layers = nn.ModuleList([])
        for i, dims in enumerate(self.dim_in_out):
            dim_in, dim_out = dims
            self.layers.append(torch.nn.Linear(dim_in, dim_out))
            if i != n_layers - 1:
                self.layers.append(torch.nn.LeakyReLU())
            else:
                print("NOT LAST LAYER ?")

    def forward(self, x):
        for h in self.layers:
            x = h(x)

        return x


def l2_loss(true_data, pred_data):
    return torch.mean((true_data - pred_data) ** 2)


def run(network_path, retrain=False, train_size=500000, batch_size=500, epochs=250):
    #brid = BrownianBridgeArbitrary(1, a=1, b=3)
    brid = BrownianBridgeArbitrary(1, a=20, b=3)
    if retrain is True:
        # Training set
        start_times, end_times, times, dataset_start, dataset_end = generate_dataset(train_size)

        perturbed_dataset = brid.sample_bridge(times, start_times, end_times, dataset_start, dataset_end)
        plt.hist(perturbed_dataset.detach().numpy()[:, :], density=True, bins=100)
        plt.title("Perturbed dataset")
        plt.show()
        #torch.save(dataset, "data/gaussian_multiple/dataset")
        #torch.save(times, "data/gaussian_multiple/times")
        #torch.save(perturbed_dataset, "data/gaussian_multiple/perturbed_dataset")

        # Test set
        start_times_test, end_times_test, times_test, dataset_start_test, dataset_end_test = generate_dataset(5000)
        perturbed_dataset_test = brid.sample_bridge(times_test, start_times_test, end_times_test, dataset_start_test,
                                                    dataset_end_test)

        #torch.save(dataset_test, "data/gaussian_multiple/dataset_test")
        #torch.save(times_test, "data/gaussian_multiple/times_test")
        #torch.save(perturbed_dataset_test, "data/gaussian_multiple/perturbed_dataset_test")

        dataset = dataSet(dataset_end, perturbed_dataset, times)
        dataLoad = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net = Network([2, 256, 256, 256], [256, 256, 256, 1])
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)
        input_test = torch.concat([perturbed_dataset_test, times_test], dim=-1)

        net.eval()
        pred_test = net.forward(input_test)
        loss_test = l2_loss(dataset_end_test, pred_test)
        #print("EPOCH:", n_epoch)
        print("LOSS TEST", torch.sqrt(loss_test))
        torch.save(net, network_path)
        #wandb.log({"train_losses": all_losses, "test_loss": loss_test})
        net.train()
        print("\n\n\n")

        if train_size % batch_size == 0:
            n_iter = train_size // batch_size
        else:
            n_iter = train_size // batch_size + 1

        for n_epoch in range(epochs):
            data = iter(dataLoad)
            all_losses = []
            for n_batch in range(n_iter):
                data_batch, perturbed_data_batch, time_batch = next(data)
                input_batch = torch.concat([perturbed_data_batch, time_batch], dim=-1)
                pred_data = net.forward(input_batch)
                loss = l2_loss(data_batch, pred_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.detach().numpy())


            print(np.mean(all_losses))
            net.eval()
            pred_test = net.forward(input_test)
            loss_test = l2_loss(dataset_end_test, pred_test)
            print("EPOCH:", n_epoch)
            print("LOSS TEST", torch.sqrt(loss_test))
            torch.save(net, network_path)
            #wandb.log({"train_losses": all_losses, "test_loss": loss_test})
            net.train()
            print("\n\n\n")


    unet = torch.load(network_path)
    times = torch.tensor(np.linspace(1, 2, 10000), dtype=torch.float32)[1:, None]
    print(times)
    traj, test = brid.euler_maruyama(torch.zeros(10000, 1), times[:, :, None], 1, unet, t_0=torch.ones((1,1)))
    #traj, test = brid.euler_maruyama(torch.randn(10000, 1)+1, times[:, :, None], 1, unet)

    print("\n\n")
    print(np.mean(test[:, 0].detach().numpy()))
    print(np.var(test[:, 0].detach().numpy()))
    plt.hist(test[:, 0].detach().numpy(), density=True, alpha=0.5, bins=20)
    plt.hist(np.random.normal(size=10000) + 1, density=True, alpha=0.5, bins=20)
    plt.show()

    plt.boxplot(test, showfliers=False)
    plt.show()

    # np.save("trajectory.npy", traj)


if __name__ == "__main__":
    # d1 = np.load("data/gaussian/generatedData1.npy")
    # d2 = np.load("data/gaussian/generatedData2.npy")
    # plt.boxplot([d1, d2], showfliers=False)
    # plt.show()
    run("data/gaussian_multiple/unet", retrain=False,  train_size=500000, batch_size=500, epochs=10000)








