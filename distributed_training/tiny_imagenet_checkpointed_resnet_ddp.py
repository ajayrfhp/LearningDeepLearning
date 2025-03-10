import torch
import time
import d2l.torch as d2l
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from models import ResnetD2l
from data import TinyImagenetD2lDDP
import collections
from IPython import display
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP


@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    x = batch["image"]
    y = batch["label"]
    return (x.to(self.device), y.to(self.device))


@d2l.add_to_class(d2l.ProgressBoard)
def draw(self, x, y, label, every_n=1):
    """Defined in :numref:`sec_utils`"""
    global title
    if not ("rank=0" in title):
        return
    Point = collections.namedtuple("Point", ["x", "y"])
    if not hasattr(self, "raw_points"):
        self.raw_points = collections.OrderedDict()
        self.data = collections.OrderedDict()
    if label not in self.raw_points:
        self.raw_points[label] = []
        self.data[label] = []
    points = self.raw_points[label]
    line = self.data[label]
    points.append(Point(x, y))
    if len(points) != every_n:
        return
    mean = lambda x: sum(x) / len(x)
    line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
    points.clear()
    if not self.display:
        return
    d2l.use_svg_display()
    if self.fig is None:
        self.fig = d2l.plt.figure(figsize=self.figsize)
    plt_lines, labels = [], []
    for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
        plt_lines.append(
            d2l.plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[
                0
            ]
        )
        labels.append(k)
    axes = self.axes if self.axes else d2l.plt.gca()
    if self.xlim:
        axes.set_xlim(self.xlim)
    if self.ylim:
        axes.set_ylim(self.ylim)
    if not self.xlabel:
        self.xlabel = self.x
    axes.set_xlabel(self.xlabel)
    axes.set_ylabel(self.ylabel)
    axes.set_xscale(self.xscale)
    axes.set_yscale(self.yscale)
    axes.legend(plt_lines, labels)
    display.display(self.fig)
    # Save the figure
    d2l.plt.savefig(f"./logs/result_{title}.png", bbox_inches="tight")
    display.clear_output(wait=True)


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
    device = torch.device("cuda", rank)

    data = TinyImagenetD2lDDP(batch_size, num_workers, is_toy=is_toy)
    num_training_batches = len(data.train_data)
    num_val_batches = len(data.val_data)
    model = ResnetD2l(
        num_classes=data.num_classes,
        pretrained=False,
        lr=learning_rate,
        checkpointed=True,
    )
    model.net.to(rank)
    model.net = DDP(model.net, device_ids=[rank], find_unused_parameters=True)

    print(f"rank={rank} model: {model.net}")
    trainer = d2l.Trainer(max_epochs=num_epochs, num_gpus=world_size)
    trainer.device = device
    trainer.fit(model=model, data=data)

    if rank == 0:
        model.display_metrics(num_training_batches, num_val_batches)

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
