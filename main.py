import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from bridge import BrownianBridge
from utils import SinusoidalPositionEmbeddings
#from denoising_diffusion_pytorch import Unet
import matplotlib.pyplot as plt
from unet import Unet


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    SIZE_training = 60000
    SIZE_test = 10000
    batch_size = 600
    mnist_dataset = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor())
    full_data_loader = DataLoader(mnist_dataset, batch_size=SIZE_training, shuffle=False)
    full_data_load_iter = iter(full_data_loader)
    full_dataset, target = next(full_data_load_iter)
    full_dataset = torch.flatten(full_dataset[target == 0, 0, :, :], start_dim=-2, end_dim=-1)
    print(full_dataset.shape)

    mnist_dataset_test = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor(), train=False)
    full_data_loader_test = DataLoader(mnist_dataset_test, batch_size=10000, shuffle=False)
    full_data_load_test_iter = iter(full_data_loader_test)
    full_dataset_test, target_test = next(full_data_load_test_iter)
    full_dataset_test = torch.flatten(full_dataset_test[target_test == 0, 0, :, :], start_dim=-2, end_dim=-1)
    print(full_dataset_test.shape)

    avg_7 = torch.reshape(torch.mean(full_dataset, dim=0), (28, 28))
    plt.imshow(avg_7.detach().numpy(), cmap="gray")
    plt.show()

    brid = BrownianBridge(784)
    times = torch.rand((5923, 1))
    perturbed_all_images = brid.sample(times, torch.zeros_like(full_dataset), full_dataset)

    times_test = torch.rand((980, 1))
    #### BE CAREFUL WE INITIALIZE AT GAUSSIAN DISTRIBUTION HERE !!!
    perturbed_all_images_test = brid.sample(times_test, torch.randn_like(full_dataset_test), full_dataset_test)


    unet = Unet(dim=8, dim_mults=(1, 2, 4), channels=1, resnet_block_groups = 8)
    #for i in range(60):
    #    print(i)
    #    batch_pertubed = perturbed_all_images[i*1000:(i+1)*1000]
    #    #batch_times_embedding = times_embedding[i*1000:(i+1)*1000]
    #    batch_times = times[i*1000:(i+1)*1000]
    #    pred = unet.forward(torch.reshape(batch_pertubed, (1000, 28, 28))[:, None, :, :], batch_times[:, 0])

    torch.save(perturbed_all_images, "data/perturbed_images")
    torch.save(times, "data/times")
    torch.save(full_dataset, "data/images")

    torch.save(perturbed_all_images_test, "data/perturbed_images_test")
    torch.save(times_test, "data/times_test")
    torch.save(full_dataset_test, "data/images_test")

    #for i in range(10000):
    #    plt.imshow(torch.reshape(perturbed_all_images[i], (28, 28)), cmap="gray")
    #    plt.show()
    #    plt.imshow(torch.reshape(full_dataset_test[i], (28, 28)), cmap="gray")
    #    plt.show()
    #    plt.imshow(torch.reshape(full_dataset[i], (28, 28)), cmap="gray")
    #    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
