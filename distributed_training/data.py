import torch
import d2l.torch as d2l
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import requests


class TinyImagenetD2l(d2l.Module):
    def __init__(self, batch_size, num_workers, is_toy=False):
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
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=train)
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
