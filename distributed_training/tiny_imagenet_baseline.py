from data import TinyImagenetD2l
from models import ResnetD2l
import d2l.torch as d2l
import matplotlib.pyplot as plt
import numpy as np


@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    x = batch["image"]
    y = batch["label"]
    return (x.to(self.device), y.to(self.device))


@d2l.add_to_class(d2l.Classifier)
def plot(self, key, value, train):
    """Plot a point in animation."""
    assert hasattr(self, "trainer"), "Trainer is not inited"

    key = ("train_" if train else "val_") + key

    value = value.item()

    if train:
        ax = self.ax[0]
    else:
        ax = self.ax[1]

    if key not in self.tracker_dict:
        self.tracker_dict[key] = [value]
        return

    self.tracker_dict[key].append(value)
    ax.clear()

    for k, v in self.tracker_dict.items():
        x_axis = np.linspace(0, len(v), len(v))
        ax.plot(
            x_axis,
            v,
            label=k,
        )
    plt.legend(loc="upper right")
    plt.pause(0.01)


def main():

    learning_rate = 0.1
    num_epochs = 10
    device = d2l.try_gpu()

    tiny_imagenet = TinyImagenetD2l(batch_size=64, num_workers=4, is_toy=True)
    train_loader = tiny_imagenet.train_dataloader()

    model = ResnetD2l(
        num_classes=tiny_imagenet.num_classes, pretrained=False, lr=learning_rate
    )
    model.to(device)
    model.tracker_dict = {}
    _, ax = plt.subplots(1, 2)
    model.ax = ax
    trainer = d2l.Trainer(max_epochs=num_epochs, num_gpus=1)
    trainer.device = device
    trainer.fit(model=model, data=tiny_imagenet)
    plt.show()


if __name__ == "__main__":
    main()
