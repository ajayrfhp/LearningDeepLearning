import torch
import d2l.torch as d2l
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import requests
from torch.utils.data.distributed import DistributedSampler
import torchvision


class TinyImageNetTorchPrimitive(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


class TinyImageNetTorch:
    def __init__(
        self,
        batch_size,
        num_workers,
        is_toy=False,
        world_size=1,
        rank=0,
        is_ddp=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_toy = is_toy
        self.world_size = world_size
        self.rank = rank
        self.is_ddp = is_ddp
        self.train_dataloader, self.val_dataloader, self.train_data, self.val_data = (
            self.get_tiny_imagenet_torch_data(
                batch_size=batch_size,
                num_workers=num_workers,
                is_ddp=is_ddp,
                rank=rank,
                world_size=world_size,
                is_toy=is_toy,
            )
        )
        self.get_imagenet_labels()

    def get_toydataset(self, dataset, num_labels=2):
        return dataset.filter(lambda x: x["label"] in list(range(num_labels)))

    def get_tiny_imagenet_torch_data(
        self,
        batch_size=1500,
        num_workers=2,
        is_ddp=False,
        rank=0,
        world_size=1,
        is_toy=False,
    ):
        transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )

        tiny_imagenet_train = load_dataset("Maysee/tiny-imagenet", split="train")
        tiny_imagenet_val = load_dataset("Maysee/tiny-imagenet", split="valid")
        self.num_classes = tiny_imagenet_train.features["label"].num_classes

        if is_toy:
            tiny_imagenet_train = self.get_toydataset(tiny_imagenet_train)
            tiny_imagenet_val = self.get_toydataset(tiny_imagenet_val)

        tiny_imagenet_train_torch = TinyImageNetTorchPrimitive(
            tiny_imagenet_train, transform=transform
        )
        tiny_imagenet_val_torch = TinyImageNetTorchPrimitive(
            tiny_imagenet_val, transform=transform
        )
        sampler = None
        if is_ddp:
            sampler = DistributedSampler(
                tiny_imagenet_train_torch, num_replicas=world_size, rank=rank
            )

        train_dataloader = torch.utils.data.DataLoader(
            tiny_imagenet_train_torch,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
        )

        val_dataloader = torch.utils.data.DataLoader(
            tiny_imagenet_val_torch,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=None,
            shuffle=False,
        )

        return train_dataloader, val_dataloader, tiny_imagenet_train, tiny_imagenet_val

    def get_imagenet_labels(self):
        response = requests.get(
            "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
            timeout=50,
        )
        imagenet_index = json.loads(response.text)
        self.imagenet_reverse_index = {v[0]: v[1] for k, v in imagenet_index.items()}

    def visualize(self, batch, max_images=16, output_path="./tmp/"):
        images, labels = batch
        images = images[:max_images]
        labels = labels[:max_images]
        images = images.permute(0, 2, 3, 1)
        labels = labels.unsqueeze(1)
        labels = self.train_data.features["label"].int2str(labels)

        labels = [
            (
                self.imagenet_reverse_index[label]
                if label in self.imagenet_reverse_index.keys()
                else label
            )
            for label in labels
        ]

        for image, label in zip(images, labels):
            random_int = torch.randint(0, 1000, (1,)).item()
            d2l.plt.imshow(image)
            d2l.plt.title(label)
            d2l.plt.savefig(f"{output_path}{label}_{random_int}.png")
            d2l.plt.show()


def get_mnist_data():
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1500, shuffle=True, num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1500, shuffle=False, num_workers=2
    )
    return train_dataloader, val_dataloader
