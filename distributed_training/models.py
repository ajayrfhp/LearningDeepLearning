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

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        val_loss = self.loss(Y_hat, batch[-1])
        val_acc = self.accuracy(Y_hat, batch[-1])
        self.metrics["loss"]["val"].append(val_loss.item())
        self.metrics["accuracy"]["val"].append(val_acc.item())
        self.plot("loss", val_loss, train=False)
        self.plot("accuracy", val_acc, train=False)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        acc = self.accuracy(self(*batch[:-1]), batch[-1])
        self.metrics["loss"]["train"].append(l.item())
        self.metrics["accuracy"]["train"].append(acc.item())
        self.plot("loss", l, train=True)
        self.plot("accuracy", acc, train=True)
        return l

    def display_metrics(self, num_training_batches, num_val_batches):
        for key, value in self.metrics.items():
            train_metric = self.get_running_mean(value["train"], num_training_batches)
            val_metric = self.get_running_mean(value["val"], num_val_batches)
            print(f"{key} - train: {train_metric}, val: {val_metric}")

    def get_running_mean(self, values, num_batches):
        return np.mean(values[-num_batches:])
