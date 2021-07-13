import torch
import torchvision.datasets

DATASET_PATH = './data'

class Dataset():
    def __init__(self, dataset, train=True, transform=None, target_transform=None):
        self.dataset = dataset(root=DATASET_PATH, train=train, transform=transform, target_transform=target_transform, download=True)

    def get_loader(self, batch_size, num_workers=1, shuffle=False):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class MNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(torchvision.datasets.MNIST, train, transform, target_transform)

class FashionMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(torchvision.datasets.FashionMNIST, train, transform, target_transform)

class CIFAR10(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(torchvision.datasets.CIFAR10, train, transform, target_transform)

class CIFAR100(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(torchvision.datasets.CIFAR100, train, transform, target_transform)