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
from torch.profiler import record_function


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

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_torch = TinyImageNet(tiny_imagenet, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_torch, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, val_loader


def get_ddp_data(rank, world_size):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")
    tiny_imagenet_torch = TinyImageNet(tiny_imagenet, transform=transform)

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
    break_after_num_batches=None,
    device=0,
    title="",
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    total_times = []
    torch.cuda.memory._record_memory_history(max_entries=10000)
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
        for epoch in range(epochs):
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
                end_time = time.time()
                total_times.append(end_time - start_time)
                if (
                    break_after_num_batches is not None
                    and batch_idx >= break_after_num_batches
                ):
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
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")

    prof.export_memory_timeline(f"{title}_memory.html", device="cuda:0")
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
