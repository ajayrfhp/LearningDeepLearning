import time
import numpy as np
import torch
from torchvision import transforms, models
from datasets import load_dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import get_data, fit_profile


if __name__ == "__main__":
    # Setup
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1500)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--num_gpus", type=int, default=1)

    args = args.parse_args()

    torch.manual_seed(710)
    np.random.seed(710)

    start_time = time.time()
    # get data
    train_loader, val_loader = get_data(args.batch_size, args.num_workers)

    # define model
    model = models.resnet18(pretrained=False)

    device = torch.device("cpu")
    if args.num_gpus == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # train model
    fit_profile(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        title=f"baseline_resnet_batch_size_{args.batch_size}",
    )

    # get time stats
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
