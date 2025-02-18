import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.profiler import record_function
from datasets import load_dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
from torch.utils.checkpoint import checkpoint


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


class ResnetCheckpointed(nn.Module):
    def __init__(self):
        super(ResnetCheckpointed, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Store individual layers
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x):
        # Apply checkpointing to each layer
        x = checkpoint(self.conv1, x)
        x = checkpoint(self.bn1, x)
        x = self.relu(x)  # ReLU is memory-efficient, no need to checkpoint
        x = checkpoint(self.maxpool, x)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        x = checkpoint(self.layer4, x)
        x = checkpoint(self.avgpool, x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_data(rank, world_size):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_torch = TinyImageNet(tiny_imagenet, transform=transform)
    num_classes = len(tiny_imagenet.features["label"].names)

    train_sampler = DistributedSampler(
        tiny_imagenet_torch, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(
        tiny_imagenet_torch, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch, batch_size=1500, num_workers=2, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch, batch_size=1500, num_workers=2, sampler=val_sampler
    )

    return train_loader, val_loader


def fit(
    model,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    break_after_num_batches=None,
    title="",
    device=0,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    total_times = []

    for epoch in range(epochs):
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()
            total_times.append(end_time - start_time)
            start_time = time.time()
    batch_ids = np.arange(len(total_times))

    total_times = np.array(total_times)
    mean_time = np.round(np.mean(total_times), 2)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(batch_ids, total_times, label=f"Mean time: {mean_time} s")
    plt.xlabel("Batch ID")
    plt.ylabel("Time (s)")
    plt.title(f"Training performance with gradient checkpointing")
    plt.legend(loc="upper right")
    plt.savefig(f"{device}_time.png")
    plt.show()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)
    model = ResnetCheckpointed()
    model.to(rank)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    train_loader, val_loader = get_data(rank, world_size)
    fit(
        model,
        train_loader,
        val_loader,
        epochs=1,
        lr=0.001,
        break_after_num_batches=None,
        title=f"checkpointed_ddp_resnet_{rank}",
        device=rank,
    )
    cleanup()


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(710)
    np.random.seed(710)

    start_time = time.time()
    mp.spawn(train, args=(2,), nprocs=2, join=True)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")
