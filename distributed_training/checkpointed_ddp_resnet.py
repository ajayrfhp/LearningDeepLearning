import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.profiler import record_function
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
from utils import get_ddp_data, fit_profile
from models import ResnetCheckpointed
import argparse


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, batch_size, num_workers):
    setup(rank, world_size)
    model = ResnetCheckpointed()
    model.to(rank)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    train_loader, val_loader = get_ddp_data(rank, world_size, batch_size, num_workers)
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

    # get batchsize, num_workers from args
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1500)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--num_gpus", type=int, default=2)
    args = args.parse_args()

    start_time = time.time()
    mp.spawn(
        train,
        args=(args.num_gpus, args.batch_size, args.num_workers),
        nprocs=args.num_gpus,
        join=True,
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")
