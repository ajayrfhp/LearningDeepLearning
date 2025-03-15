import torch
import time
import d2l.torch as d2l
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from d2l_data import TinyImagenetD2l
from models import ResnetD2l
from trainer import CustomTrainer


def main():
    batch_size = 256
    num_workers = 2
    learning_rate = 0.01
    num_epochs = 20
    device = d2l.try_gpu()

    data = TinyImagenetD2l(batch_size, num_workers, is_toy=True)
    model = ResnetD2l(num_classes=data.num_classes, pretrained=False, lr=learning_rate)
    model.to(device)

    trainer = CustomTrainer(
        max_epochs=num_epochs,
        num_gpus=1,
        title=f"TinyImagenet_is_toy_{data.is_toy}_lr_{learning_rate}",
        plot_every_n_steps=2,
    )
    trainer.device = device
    trainer.fit(model=model, data=data)


if __name__ == "__main__":
    main()
