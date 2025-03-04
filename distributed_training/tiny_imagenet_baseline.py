import torch
import time
import d2l.torch as d2l
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchvision.models import resnet18
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


@d2l.add_to_class(d2l.Classifier)
def validation_step(self, batch):
    Y_hat = self(*batch[:-1])
    val_loss = self.loss(Y_hat, batch[-1])
    val_acc = self.accuracy(Y_hat, batch[-1])
    self.metrics["loss"]["val"].append(val_loss.item())
    self.metrics["accuracy"]["val"].append(val_acc.item())
    self.plot("loss", val_loss, train=False)
    self.plot("accuracy", val_acc, train=False)


@d2l.add_to_class(d2l.Classifier)
def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    acc = self.accuracy(self(*batch[:-1]), batch[-1])
    self.metrics["loss"]["train"].append(l.item())
    self.metrics["accuracy"]["train"].append(acc.item())
    self.plot("loss", l, train=True)
    self.plot("accuracy", acc, train=True)
    return l


@d2l.add_to_class(d2l.Classifier)
def display_metrics(self, num_training_batches, num_val_batches):
    for key, value in self.metrics.items():
        train_metric = self.get_running_mean(value["train"], num_training_batches)
        val_metric = self.get_running_mean(value["val"], num_val_batches)
        print(f"{key} - train: {train_metric}, val: {val_metric}")


@d2l.add_to_class(d2l.Classifier)
def get_running_mean(self, values, num_batches):
    return np.mean(values[-num_batches:])


class ResnetD2l(d2l.Classifier):
    def __init__(self, num_classes, pretrained=False, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = resnet18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, num_classes)
        self.board = d2l.ProgressBoard()
        self.metrics = defaultdict(
            lambda: {"train": [], "val": [], "figure": None, "subplot": None}
        )
        self.figsize = (10, 5)

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return nn.CrossEntropyLoss()(y_hat, y)


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
    display.clear_output(wait=True)

    # Save the figure
    d2l.plt.savefig(f"./tmp/result.png")


def main():
    batch_size = 256
    num_workers = 2
    learning_rate = 0.01
    num_epochs = 20
    device = d2l.try_gpu()

    data = TinyImagenetD2l(batch_size, num_workers, is_toy=True)
    num_training_batches = len(data.train_data)
    num_val_batches = len(data.val_data)
    model = ResnetD2l(num_classes=data.num_classes, pretrained=False, lr=learning_rate)
    model.to(device)

    trainer = d2l.Trainer(max_epochs=num_epochs, num_gpus=1)
    trainer.device = device
    trainer.fit(model=model, data=data)

    model.display_metrics(num_training_batches, num_val_batches)


if __name__ == "__main__":
    main()
