import time
import numpy as np
import torch
from torchvision import transforms, models
from datasets import load_dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


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


def get_data(batch_size, num_workers):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    tiny_imagenet_train = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_val = load_dataset("Maysee/tiny-imagenet", split="valid")
    tiny_imagenet_torch_train = TinyImageNet(tiny_imagenet_train, transform=transform)
    tiny_imagenet_torch_val = TinyImageNet(tiny_imagenet_val, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch_train, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch_val, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, val_loader


def get_ddp_data(rank, world_size):
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
        tiny_imagenet_train_torch, batch_size=1500, num_workers=2, sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_val_torch, batch_size=1500, num_workers=2, sampler=None
    )

    return train_loader, val_loader


def clear_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    import gc

    gc.collect()
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def fit_profile(
    model,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    num_steps=np.inf,
    device=0,
    title="",
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
            criterion(outputs, labels).backward()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()
            total_times.append(end_time - start_time)

            print(f"Training batch {batch_idx}")

            if batch_idx >= num_steps:
                break
            start_time = time.time()
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1, keepdim=True)
                val_acc += preds.eq(labels.view_as(preds)).sum().item()
                print(f"Validation batch {batch_idx}")
                if batch_idx >= num_steps:
                    break
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")

    batch_ids = np.arange(len(total_times))

    total_times = np.array(total_times)
    mean_time = np.round(np.mean(total_times), 2)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(batch_ids, total_times, label=f"Mean time: {mean_time} s")
    plt.xlabel("Batch ID")
    plt.ylabel("Time (s)")
    plt.title(f"Training performance with gradient checkpointing")
    plt.legend()
    plt.savefig(f"{title}_time.png")
    plt.show()
