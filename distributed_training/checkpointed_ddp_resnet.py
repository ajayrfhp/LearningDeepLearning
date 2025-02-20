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
from utils import get_ddp_data, fit_profile
from models import ResnetCheckpointed


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

    train_loader, val_loader = get_ddp_data(rank, world_size)
    fit_profile(
        model,
        train_loader,
        val_loader,
        epochs=1,
        lr=0.001,
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