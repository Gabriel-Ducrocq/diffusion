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
from utils import dataSet
import wandb
import time


wandb.init(
    project="diffusion-gaussian-example",
    
    config={
        "learning_rate":0.003,
        "optimizer":"adam",
        "architecure":"MLP",
        "size":[256, 256, 256, 1],
        "epochs":10
    }
)




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


def run(retrain=False, mu = 1, data_path="data/gaussian_experiment_mu100/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    #brid = BrownianBridge(1, a=1, b=3)
    brid = BrownianBridge(1, a=25, b=5)
    if retrain is True:
        #Training set
        dataset = torch.tensor(np.random.normal(size = (500000,1)) + mu, dtype=torch.float32)
        times = torch.rand((500000, 1))
        perturbed_dataset = brid.sample(times, torch.randn_like(dataset), dataset)

        torch.save(dataset, data_path + "dataset")
        torch.save(times, data_path + "times")
        torch.save(perturbed_dataset, data_path + "perturbed_dataset")

        #Test set
        dataset_test = torch.tensor(np.random.normal(size = (5000,1)) + mu, dtype=torch.float32)
        times_test = torch.rand((5000, 1))
        perturbed_dataset_test = brid.sample(times_test, torch.randn_like(dataset_test), dataset_test)

        torch.save(dataset_test, data_path + "dataset_test")
        torch.save(times_test, data_path + "times_test")
        torch.save(perturbed_dataset_test, data_path + "perturbed_dataset_test")


        batch_size = 50
        dataset = dataSet(dataset, perturbed_dataset, times)
        dataLoad = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        net = Network([2, 256, 256, 256], [256, 256, 256, 1])
        net = net.to(device)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.00003)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.00003)
        epochs = 10000
        input_test = torch.concat([perturbed_dataset_test, times_test], dim=-1)
        input_test = input_test.to(device)
        dataset_test = dataset_test.to(device)
        for n_epoch in range(epochs):
            data = iter(dataLoad)
            all_losses = []
            start = time.time()
            for n_batch in range(10000):
                data_batch, perturbed_data_batch, time_batch = next(data)
                data_batch = data_batch.to(device)
                batch_size = data_batch.shape[0]
                input_batch = torch.concat([perturbed_data_batch, time_batch], dim = -1)
                input_batch = input_batch.to(device)
                pred_data = net.forward(input_batch)
                loss = l2_loss(data_batch, pred_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.detach().cpu().numpy())

            end = time.time()
            print("Time on epoch:", end - start)
            print("Train Loss:", np.mean(all_losses))
            net.eval()
            pred_test = net.forward(input_test)
            print(pred_test)
            print(dataset_test)
            loss_test = l2_loss(dataset_test, pred_test)
            print("EPOCH:", n_epoch)
            print("LOSS TEST", torch.sqrt(loss_test))
            torch.save(net, data_path + "network")
            wandb.log({"train_losses": all_losses, "test_loss": loss_test.detach().cpu().numpy()})
            net.train()
            print("\n\n\n")


    unet = torch.load(data_path + "network", map_location=torch.device('cpu'))
    unet.eval()
    times = torch.tensor(np.linspace(0, 1, 1000), dtype=torch.float32)[:, None]
    traj, test = brid.euler_maruyama(torch.randn(10000, 1),times[:, :, None], 1, unet)

    #np.save("data/gaussian/generatedData2.npy", test[:, 0].detach().numpy())
    print("\n\n")
    print(np.mean(test[:, 0].detach().numpy()))
    print(np.var(test[:, 0].detach().numpy()))
    plt.hist(test[:, 0].detach().numpy(), density=True, alpha=0.5, bins= 20)
    plt.hist(np.random.normal(size=10000)+mu, density=True, alpha=0.5, bins=20)
    plt.show()

    plt.boxplot(test, showfliers=True)
    plt.show()

    #np.save("trajectory.npy", traj)


if __name__=="__main__":
    #d1 = np.load("data/gaussian/generatedData1.npy")
    #d2 = np.load("data/gaussian/generatedData2.npy")
    #plt.boxplot([d1, d2], showfliers=False)
    #plt.show()

    run(retrain=True, mu=100)








