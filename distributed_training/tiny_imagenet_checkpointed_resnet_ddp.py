import torch
import time
import d2l.torch as d2l
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from models import ResnetD2l
from data import get_ddp_data
import collections
from IPython import display
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import fit


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(
    rank, world_size, batch_size, num_workers, num_epochs, learning_rate, is_toy=False
):
    global title
    title = f"Tiny ImageNet Checkpointed resnet ddp Toy={is_toy} batch_size{batch_size} num_workers{num_workers} num_epochs{num_epochs} learning_rate{learning_rate} rank={rank}"
    setup(rank, world_size)

    train_loader, val_loader, train_data, _ = get_ddp_data(rank, world_size)
    model = ResnetD2l(
        num_classes=train_data.features["label"].num_classes,
        pretrained=False,
        lr=learning_rate,
        checkpointed=True,
    )
    model.net.to(rank)
    model.net = DDP(model.net, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        rank=rank,
        title=title,
    )

    cleanup()


title = None

if __name__ == "__main__":
    torch.manual_seed(710)
    np.random.seed(710)

    # get batchsize, num_workers from args
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=125)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--num_gpus", type=int, default=1)
    args.add_argument("--num_epochs", type=int, default=2)
    args.add_argument("--learning_rate", type=float, default=0.01)
    args.add_argument("--is_toy", type=bool, default=True)
    args = args.parse_args()

    start_time = time.time()
    mp.spawn(
        train,
        args=(
            args.num_gpus,
            args.batch_size,
            args.num_workers,
            args.num_epochs,
            args.learning_rate,
            args.is_toy,
        ),
        nprocs=args.num_gpus,
        join=True,
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")
