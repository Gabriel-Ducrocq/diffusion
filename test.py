import matplotlib.pyplot as plt
import torch
import numpy as np
from bridge import BrownianBridge
from gaussian_example import l2_loss
from gaussian_example import Network



brid = BrownianBridge(1, a=3, b=5)
dataset_test = torch.tensor(np.random.normal(size=(5000, 1)) + 3, dtype=torch.float32)
times_test = torch.rand((5000, 1))*0.1

perturbed_dataset_test = brid.sample(times_test, torch.randn_like(dataset_test), dataset_test)
net = torch.load("data/gaussian/unet")

input_test = torch.concat([perturbed_dataset_test, times_test], dim=-1)
pred_test = net.forward(input_test)
print(pred_test)
print(dataset_test)
loss_test = l2_loss(dataset_test, pred_test)
print("LOSS TEST", torch.sqrt(loss_test))

