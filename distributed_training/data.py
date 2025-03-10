import torch
import d2l.torch as d2l
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import requests
from torch.utils.data.distributed import DistributedSampler


class TinyImagenetD2lDDP(d2l.Module):
    def __init__(self, batch_size, num_workers, is_toy=False, world_size=1, rank=0):
        super().__init__()
        self.save_hyperparameters()
        self.train_data = load_dataset("Maysee/tiny-imagenet", split="train")
        self.val_data = load_dataset("Maysee/tiny-imagenet", split="valid")
        self.transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )
        self.get_imagenet_labels()
        if is_toy:
            self.train_data = self.get_toydataset(self.train_data)
            self.val_data = self.get_toydataset(self.val_data)

        # get number of classes
        self.num_classes = self.train_data.features["label"].num_classes
        self.sampler = DistributedSampler(
            self.train_data, num_replicas=world_size, rank=rank
        )

    def get_toydataset(self, dataset, num_labels=2):
        dataset = dataset.filter(lambda x: x["label"] in list(range(num_labels)))
        return dataset

    def get_imagenet_labels(self):
        response = requests.get(
            "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
            timeout=50,
        )
        imagenet_index = json.loads(response.text)
        self.imagenet_reverse_index = {v[0]: v[1] for k, v in imagenet_index.items()}

    def transforms(self, batch):
        batch["image"] = [self.transform(x.convert("RGB")) for x in batch["image"]]
        batch["label"] = torch.tensor(batch["label"])
        return batch

    def get_dataloader(self, train):
        data = self.train_data if train else self.val_data
        data.set_transform(self.transforms)
        sampler = self.sampler if train else None
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def visualize(self, batch, max_images=16, output_path="./tmp/"):
        images = batch["image"][:max_images].permute(0, 2, 3, 1)
        labels = batch["label"][:max_images].unsqueeze(1)
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
        [transforms.Resize((224, 224)), transforms.ToTensor()]
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
