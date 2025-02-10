import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint_sequential
from datasets import load_dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import argparse


def get_tiny_imagenet_data():
    class TinyImageNet(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]["image"], self.dataset[idx]["label"]
            # convert x to RGB
            x = x.convert("RGB")
            if self.transform:
                x = self.transform(x)
            y = torch.tensor(y, dtype=torch.int64)
            return x, y

    # Define transformations
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_torch = TinyImageNet(tiny_imagenet, transform=transform)
    return tiny_imagenet_torch


def clear_cuda_memory():
    torch.cuda.empty_cache()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Clear memory allocated by PyTorch
    torch.cuda.synchronize()

    gc.collect()


def fit_profile(
    model,
    train_loader,
    device,
    epochs=1,
    lr=0.001,
    break_after_num_batches=None,
    title="",
):
    clear_cuda_memory()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    total_times = []

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:

        for _ in range(epochs):
            start_time = time.time()
            for batch_idx, batch in enumerate(train_loader):
                prof.step()

                inputs, labels = batch

                with record_function("to_device"):
                    inputs, labels = inputs.to(device), labels.to(device)

                with record_function("forward"):
                    outputs = model(inputs)

                with record_function("backward"):
                    criterion(outputs, labels).backward()

                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()

                if (
                    break_after_num_batches is not None
                    and batch_idx >= break_after_num_batches
                ):
                    break

                end_time = time.time()
                total_times.append(end_time - start_time)
                start_time = time.time()

    prof.export_memory_timeline(f"{title}_memory.html", device="cuda:0")

    total_times = np.array(total_times)
    mean_time = np.mean(total_times)
    batch_ids = np.arange(len(total_times))

    plt.figure(figsize=(10, 5))
    plt.plot(
        batch_ids,
        total_times,
        label=f"load time avg {mean_time}",
        marker="o",
        alpha=0.5,
    )

    plt.xlabel("Batch ID")
    plt.ylabel("Time (s)")
    plt.title(f"Data load times with  avg total time {mean_time}")
    plt.legend()

    # Save the plot
    plt.savefig(f"{title}_time.png")
    plt.show()


class ResnetCheckpointed(nn.Module):
    def __init__(self):
        super(ResnetCheckpointed, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Create a sequential container for the features
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool,
        )
        self.fc = self.model.fc

        # Number of segments to split the features into for checkpointing
        self.segments = 3

    def forward(self, x):
        # Apply checkpoint_sequential to features
        x = checkpoint_sequential(self.features, self.segments, x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="resnet18")
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--break_after_num_batches", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--enable_checkpointing", type=bool, default=False)
    args.add_argument("--num_gpus", type=int, default=1)

    args = args.parse_args()

    tiny_imagenet_data = get_tiny_imagenet_data()
    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.num_gpus == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = models.resnet18(pretrained=False)

    if args.enable_checkpointing:
        del model
        model = ResnetCheckpointed()

    fit_profile(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        break_after_num_batches=args.break_after_num_batches,
        title=f"{args.model}_checkpointed_{args.enable_checkpointing}_batch_size_{args.batch_size}",
    )
