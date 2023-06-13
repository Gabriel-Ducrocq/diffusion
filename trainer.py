import torch
import numpy as np
from torch.utils.data import DataLoader


def l2_loss(true_data, pred_data):
    return torch.mean(torch.sum((true_data - pred_data)**2, dim=-1))

class Trainer:
    def __init__(self, network, optimizer, batch_size=50):
        """

        :param network: torch neural network
        :param optimizer: torch optimizer
        :param batch_size: int, batch_size
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.all_losses_train = []
        self.all_losses_test = []
        self.epochs_number = 0

    def eval(self, dataLoad_test):
        data_test, param_test, perturbed_param_test, time_test = next(iter(dataLoad_test))
        input_test = torch.concat([data_test, perturbed_param_test, time_test], dim=-1)
        input_test = input_test.to(self.device)
        param_test = param_test.to(self.device)
        self.network.eval()
        pred_test = self.network.forward(input_test)
        loss_test = l2_loss(param_test, pred_test)
        self.all_losses_test.append(loss_test.detach().cpu().numpy())
        print("EPOCH:", self.epochs_number)
        print("--LOSS TEST", loss_test)
        print("\n")

    def train(self, data_train, data_test):
        dataLoad_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        dataLoad_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False)

        self.eval(dataLoad_test)
        while True:
            self.network.train()
            dataLoad_train_iter = iter(dataLoad_train)
            for data_batch, param_batch, perturbed_param_batch, time_batch in dataLoad_train_iter:
                self.optimizer.zero_grad()
                input_batch = torch.concat([data_batch, perturbed_param_batch, time_batch], dim = -1)
                input_batch = input_batch.to(self.device)
                pred_data = self.network.forward(input_batch)
                param_batch = param_batch.to(self.device)
                loss = l2_loss(param_batch, pred_data)
                loss.backward()
                self.optimizer.step()
                self.all_losses_train.append(loss.detach().cpu().numpy())

            self.eval(dataLoad_test)
            self.epochs_number += 1






