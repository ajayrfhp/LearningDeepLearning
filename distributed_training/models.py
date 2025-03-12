import torch.nn as nn
import torch
from torchvision import transforms, models
from torch.utils.checkpoint import checkpoint
import d2l.torch as d2l
from collections import defaultdict
from torchvision.models import resnet18
import numpy as np


class ResnetCheckpointed(nn.Module):
    def __init__(self, pretrained=False):
        super(ResnetCheckpointed, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        # Store individual layers
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x):
        # Apply checkpointing to each layer
        x = checkpoint(self.conv1, x)
        x = checkpoint(self.bn1, x)
        x = self.relu(x)  # ReLU is memory-efficient, no need to checkpoint
        x = checkpoint(self.maxpool, x)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        x = checkpoint(self.layer4, x)
        x = checkpoint(self.avgpool, x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResnetD2l(d2l.Classifier):
    def __init__(self, num_classes, pretrained=False, lr=0.01, checkpointed=False):
        super().__init__()
        self.save_hyperparameters()
        if checkpointed:
            self.net = ResnetCheckpointed(pretrained=pretrained)
        else:
            self.net = resnet18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, num_classes)

        self.metrics = defaultdict(
            lambda: {
                "train": [],
                "val": [],
                "figure": None,
                "ax": None,
            }
        )

        self.aggregate_metrics = []

        self.figsize = (10, 5)

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return nn.CrossEntropyLoss()(y_hat, y)

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        val_loss = self.loss(Y_hat, batch[-1])
        val_acc = self.accuracy(Y_hat, batch[-1])
        self.trainer.val_batch_idx += 1
        self.metrics["loss"]["val"].append(val_loss.item())
        self.metrics["accuracy"]["val"].append(val_acc.item())
        if self.trainer.train_batch_idx % self.trainer.plot_every_n_steps == 0:
            self.update_aggregate_metrics()
            self.update_plot()

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        acc = self.accuracy(self(*batch[:-1]), batch[-1])
        self.trainer.train_batch_idx += 1
        self.metrics["loss"]["train"].append(l.item())
        self.metrics["accuracy"]["train"].append(acc.item())

        if self.trainer.train_batch_idx % self.trainer.plot_every_n_steps == 0:
            self.update_aggregate_metrics()
            self.update_plot()
        return l

    def update_aggregate_metrics(self):
        # update the aggregate metrics with values from self.metrics for current_epoch
        aggregate_metric = {}
        for metric, value in self.metrics.items():
            train_agg = self.get_running_mean(value["train"])
            val_agg = self.get_running_mean(value["val"])
            aggregate_metric[metric] = {"train": train_agg, "val": val_agg}

        if len(self.aggregate_metrics) <= self.trainer.epoch:
            self.aggregate_metrics.append(aggregate_metric)
        else:
            self.aggregate_metrics[self.trainer.epoch] = aggregate_metric

    def display_metrics(self):
        latest_metrics = self.aggregate_metrics[self.trainer.epoch - 1]
        for key, value in latest_metrics.items():
            print(f"""{key} - train: {value["train"]}, val: {value["val"]}""")

    def get_running_mean(self, values, num_batches=0):
        return np.mean(values[-num_batches:])

    def update_plot(self):

        for metric in self.metrics.keys():
            ax = self.metrics[metric]["ax"]
            figure = self.metrics[metric]["figure"]

            if figure is None or ax is None:
                figure, ax = d2l.plt.subplots(figsize=self.figsize)
                self.metrics[metric]["figure"] = figure
                self.metrics[metric]["ax"] = ax

            train_values, val_values, x_values = [], [], []

            # iterate over all epochs and get train and val values
            for epoch, metric_values in enumerate(self.aggregate_metrics):
                train_values.append(metric_values[metric]["train"])
                val_values.append(metric_values[metric]["val"])
                x_values.append(epoch)

            # Clear the current figure
            ax.clear()
            ax.plot(x_values, train_values, label="train")
            ax.plot(x_values, val_values, label="val")
            ax.set_xlabel("num_epochs")
            ax.set_ylabel(metric)
            ax.legend()
            figure.savefig(f"./logs/{self.trainer.title}_{metric}.png")
            figure.show()
            d2l.plt.close(figure)
