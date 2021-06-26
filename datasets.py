import torch
import torchvision.datasets

class MNIST():
    def __init__(self, train=True, transform=None):
        self.dataset = torchvision.datasets.MNIST(root='./datasets', train=train, transform=transform, download=True)

    def get_loader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

class FashionMNIST():
    def __init__(self, train=True, transform=None):
        self.dataset = torchvision.datasets.FashionMNIST(root='./datasets', train=train, transform=transform, download=True)

    def get_loader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
