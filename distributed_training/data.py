import torch
import d2l.torch as d2l
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import requests
from torch.utils.data.distributed import DistributedSampler
import torchvision


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]["image"], self.dataset[idx]["label"]
        x = x.convert("RGB")
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(y, dtype=torch.int64)
        return x, y


def get_ddp_data(rank, world_size, batch_size=1500, num_workers=2):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    tiny_imagenet_train = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_val = load_dataset("Maysee/tiny-imagenet", split="valid")

    tiny_imagenet_train_torch = TinyImageNet(tiny_imagenet_train, transform=transform)
    tiny_imagenet_val_torch = TinyImageNet(tiny_imagenet_val, transform=transform)

    train_sampler = DistributedSampler(
        tiny_imagenet_train_torch, num_replicas=world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_train_torch,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_val_torch,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )

    return train_loader, val_loader, tiny_imagenet_train, tiny_imagenet_val


def get_mnist_data():
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1500, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1500, shuffle=False, num_workers=2
    )
    return train_loader, val_loader
