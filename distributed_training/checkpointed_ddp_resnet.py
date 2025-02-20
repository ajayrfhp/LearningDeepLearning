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
import argparse

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, args):
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
        epochs=args.epochs,
        lr=args.lr,
        device=rank,
        num_steps=args.num_steps,
        title=f"baseline_resnet_batch_size_{args.batch_size}_checkpointing_{args.enable_checkpointing}_num_gpus_{args.num_gpus}_ddp",
    )

    cleanup()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1500)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--enable_checkpointing", type=bool, default=False)
    args.add_argument("--num_gpus", type=int, default=1)
    args.add_argument("--num_steps", type=int, default=np.inf)
    args.add_argument("--num_workers", type=int, default=2)

    args = args.parse_args()

    torch.manual_seed(710)
    np.random.seed(710)


    start_time = time.time()
    mp.spawn(train, args=(2, args), nprocs=2, join=True)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")