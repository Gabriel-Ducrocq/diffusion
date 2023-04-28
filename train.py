import matplotlib.pyplot as plt
import torch
from unet import Unet
from utils import dataSet, l2_loss
from torch.utils.data import DataLoader



batch_size = 600
#batch_size = 2
data_path = "data"

times = torch.load("data/times")
perturbed_images = torch.load("data/perturbed_images")
images = torch.load("data/images")
dataset = dataSet(images, perturbed_images, times)
dataLoad = DataLoader(dataset, batch_size=batch_size, shuffle=True)
unet = Unet(dim=8, dim_mults=(1, 2, 4), channels=1)

optimizer = torch.optim.Adam(unet.parameters(), lr=0.0003)
epochs = 1000

for n_epoch in range(epochs):
    data = iter(dataLoad)
    for n_batch in range(10):
        print(n_batch/10)
        images_batch, perturbed_images_batch, time_batch = next(data)
        batch_size = images_batch.shape[0]
        images_pred = unet.forward(torch.reshape(perturbed_images_batch, (batch_size, 28, 28))[:, None, :, :],
                                   time_batch[:, 0])
        loss = l2_loss(torch.reshape(images_batch, (batch_size, 28, 28))[:, None, :, :], images_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("Loss:", loss)

    times_test = torch.load("data/times_test")
    perturbed_images_test = torch.load("data/perturbed_images_test")
    images_test = torch.load("data/images_test")
    images_pred_test = unet.forward(torch.reshape(perturbed_images_test, (980, 28, 28))[:, None, :, :],
                               times_test[:, 0])
    loss_test = l2_loss(torch.reshape(images_test, (980, 28, 28)), images_pred_test)
    print("EPOCH:", n_epoch)
    print("LOSS TEST", loss_test)
    torch.save(unet, "data/unet")
    print("\n\n\n")













