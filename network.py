import torch
from torch import nn


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
                #self.layers.append(torch.nn.SiLU())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                print("NOT LAST LAYER ?")

    def forward(self, x):
        for h in self.layers:
            x = h(x)

        return x