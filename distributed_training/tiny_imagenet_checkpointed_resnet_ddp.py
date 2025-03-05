import torch
import time
import d2l.torch as d2l
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from models import ResnetD2l
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Image
import json
import requests
from data import TinyImagenetD2l
import collections
from IPython import display
from collections import defaultdict


@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    x = batch["image"]
    y = batch["label"]
    return (x.to(self.device), y.to(self.device))


@d2l.add_to_class(d2l.ProgressBoard)
def draw(self, x, y, label, every_n=1):
    """Defined in :numref:`sec_utils`"""
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
    d2l.plt.savefig(f"./tmp/result_{title}.png", bbox_inches="tight")
    display.clear_output(wait=True)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, batch_size, num_workers, num_epochs, learning_rate):
    setup(rank, world_size)

    device = d2l.try_gpu()

    data = TinyImagenetD2l(batch_size, num_workers, is_toy=False)
    num_training_batches = len(data.train_data)
    num_val_batches = len(data.val_data)
    model = ResnetD2l(num_classes=data.num_classes, pretrained=False, lr=learning_rate)
    model.to(device)

    trainer = d2l.Trainer(max_epochs=num_epochs, num_gpus=1)
    trainer.device = device
    trainer.fit(model=model, data=data)

    model.display_metrics(num_training_batches, num_val_batches)

    cleanup()


if __name__ == "__main__":
    torch.manual_seed(710)
    np.random.seed(710)

    # get batchsize, num_workers from args
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1500)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--num_gpus", type=int, default=2)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--learning_rate", type=float, default=0.01)
    args = args.parse_args()

    start_time = time.time()
    title = "Tiny ImageNet Checkpointed resnet ddp"
    mp.spawn(
        train,
        args=(args.num_gpus, args.batch_size, args.num_workers, args.num_epochs, args.learning_rate),
        nprocs=args.num_gpus,
        join=True,
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")
