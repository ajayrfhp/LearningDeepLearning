import torch.nn as nn
import torch
from torchvision import transforms, models
from torch.utils.checkpoint import checkpoint


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
