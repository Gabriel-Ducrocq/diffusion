import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from bridge import BrownianBridge
from utils import SinusoidalPositionEmbeddings
#from denoising_diffusion_pytorch import Unet
import matplotlib.pyplot as plt
from unet import Unet
from utils import dataSet, l2_loss
from torch.utils.data import DataLoader
import numpy as np


def test(retrain=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    if retrain:
        SIZE_training = 60000
        SIZE_test = 10000
        batch_size = 600
        mnist_dataset = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor())
        full_data_loader = DataLoader(mnist_dataset, batch_size=SIZE_training, shuffle=False)
        full_data_load_iter = iter(full_data_loader)
        full_dataset, target = next(full_data_load_iter)
        full_dataset = torch.flatten(full_dataset[target == 0, 0, :, :], start_dim=-2, end_dim=-1)
        print(full_dataset.shape)

        full_end_dataset = torch.clone(full_dataset)
        full_dataset[:, :392] = 0

        mnist_dataset_test = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor(), train=False)
        full_data_loader_test = DataLoader(mnist_dataset_test, batch_size=10000, shuffle=False)
        full_data_load_test_iter = iter(full_data_loader_test)
        full_dataset_test, target_test = next(full_data_load_test_iter)
        full_dataset_test = torch.flatten(full_dataset_test[target_test == 0, 0, :, :], start_dim=-2, end_dim=-1)
        print(full_dataset_test.shape)

        full_end_dataset_test = torch.clone(full_dataset_test)
        full_dataset_test[:, :392] = 0

        avg_7 = torch.reshape(torch.mean(full_dataset, dim=0), (28, 28))
        plt.imshow(avg_7.detach().numpy(), cmap="gray")
        plt.title("Start")
        plt.show()

        avg_7 = torch.reshape(torch.mean(full_end_dataset, dim=0), (28, 28))
        plt.imshow(avg_7.detach().numpy(), cmap="gray")
        plt.title("End")
        plt.show()

        brid = BrownianBridge(784, a=1, use_unet=True)
        times = torch.rand((5923, 1))
        perturbed_all_images = brid.sample(times, full_dataset, full_end_dataset)

        times_test = torch.rand((980, 1))
        #### BE CAREFUL WE INITIALIZE AT GAUSSIAN DISTRIBUTION HERE !!!
        perturbed_all_images_test = brid.sample(times_test, full_dataset_test, full_end_dataset_test)


        unet = Unet(dim=8, dim_mults=(1, 2, 4), channels=1, resnet_block_groups = 8)
        #for i in range(60):
        #    print(i)
        #    batch_pertubed = perturbed_all_images[i*1000:(i+1)*1000]
        #    #batch_times_embedding = times_embedding[i*1000:(i+1)*1000]
        #    batch_times = times[i*1000:(i+1)*1000]
        #    pred = unet.forward(torch.reshape(batch_pertubed, (1000, 28, 28))[:, None, :, :], batch_times[:, 0])

        torch.save(perturbed_all_images, "../data/inpainting/perturbed_images")
        torch.save(times, "../data/inpainting/times")
        torch.save(full_dataset, "../data/inpainting/images_start")
        torch.save(full_end_dataset, "../data/inpainting/images_end")

        torch.save(perturbed_all_images_test, "../data/inpainting/perturbed_images_test")
        torch.save(times_test, "../data/inpainting/times_test")
        torch.save(full_dataset_test, "../data/inpainting/images_start_test")
        torch.save(full_end_dataset_test, "../data/inpainting/images_end_test")

        batch_size = 600
        # batch_size = 2
        data_path = "data"

        times = torch.load("../data/inpainting/times")
        perturbed_images = torch.load("../data/inpainting/perturbed_images")
        images = torch.load("../data/inpainting/images_end")
        dataset = dataSet(images, perturbed_images, times)
        dataLoad = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        unet = Unet(dim=8, dim_mults=(1, 2, 4), channels=1)

        optimizer = torch.optim.Adam(unet.parameters(), lr=0.0003)
        epochs = 1000

        for n_epoch in range(epochs):
            data = iter(dataLoad)
            for n_batch in range(10):
                print(n_batch / 10)
                images_batch, perturbed_images_batch, time_batch = next(data)
                batch_size = images_batch.shape[0]
                images_pred = unet.forward(torch.reshape(perturbed_images_batch, (batch_size, 28, 28))[:, None, :, :],
                                           time_batch[:, 0])
                loss = l2_loss(torch.reshape(images_batch, (batch_size, 28, 28))[:, None, :, :], images_pred)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print("Loss:", loss)

            times_test = torch.load("../data/inpainting/times_test")
            perturbed_images_test = torch.load("../data/inpainting/perturbed_images_test")
            images_test = torch.load("../data/inpainting/images_end_test")
            images_pred_test = unet.forward(torch.reshape(perturbed_images_test, (980, 28, 28))[:, None, :, :],
                                            times_test[:, 0])
            loss_test = l2_loss(torch.reshape(images_test, (980, 28, 28)), images_pred_test)
            print("EPOCH:", n_epoch)
            print("LOSS TEST", loss_test)
            torch.save(unet, "../data/inpainting/unet")
            print("\n\n\n")

        #for i in range(10000):
        #    plt.imshow(torch.reshape(perturbed_all_images[i], (28, 28)), cmap="gray")
        #    plt.show()
        #    plt.imshow(torch.reshape(full_dataset_test[i], (28, 28)), cmap="gray")
        #    plt.show()
        #    plt.imshow(torch.reshape(full_dataset[i], (28, 28)), cmap="gray")
        #    plt.show()


    bB = BrownianBridge(28*28, use_unet=True, a=1)
    unet = torch.load("../data/inpainting/unet")
    times = torch.tensor(np.linspace(0, 1, 10000), dtype=torch.float32)[:, None, None]

    images_start_test = torch.load("../data/inpainting/images_start_test")
    #images_start_test = torch.reshape(images_start_test, (980, 28, 28))
    print(images_start_test.shape)
    traj, test = bB.euler_maruyama(images_start_test[2][None, :], times[1:, :, :], 1, unet)
    #traj, test = bB.euler_maruyama(torch.zeros_like(images_start_test[0][None, :], dtype=torch.float32), times[1:, :, :], 1, unet)
    np.save("trajectory.npy", traj)

    image_start = torch.reshape(images_start_test[2], (28, 28))
    plt.imshow(image_start.detach().numpy(), cmap="gray")
    plt.show()
    test = torch.reshape(test[0], (28, 28))
    test = torch.max(test, torch.zeros_like(test))
    test = torch.min(test, torch.ones_like(test))
    test = test/torch.max(test)
    print(test)
    plt.imshow(test.detach().numpy(), cmap="gray")
    plt.show()


    image_diff = test - torch.reshape(images_start_test[2], (28, 28))
    plt.imshow(image_diff.detach().numpy(), cmap="gray")
    plt.show()

    image_diff = test - torch.reshape(images_start_test[0], (28, 28))
    plt.imshow(image_diff.detach().numpy(), cmap="gray")
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test(retrain=False)