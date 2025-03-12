# train mnist against our utils

from utils import fit
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torchvision import transforms
from data import get_mnist_data

rank = 0
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = get_mnist_data()
model = nn.Sequential(
    *[
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
    ]
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model.to(device="cuda")

fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=10,
    log_interval=100,
)
