import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import load_dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import get_data, fit_profile
from models import ResnetCheckpointed


if __name__ == "__main__":
    # Setup
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1500)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--enable_checkpointing", type=bool, default=False)
    args.add_argument("--num_gpus", type=int, default=1)
    args.add_argument("--num_steps", type=int, default=np.inf)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--enable_data_parallel", type=bool, default=False)

    args = args.parse_args()

    torch.manual_seed(710)
    np.random.seed(710)

    start_time = time.time()
    # get data
    train_loader, val_loader = get_data(args.batch_size, args.num_workers)

    model = None
    if args.enable_checkpointing:
        model = ResnetCheckpointed()
    else:
        model = models.resnet18(pretrained=False)

    device = torch.device("cpu")

    if args.num_gpus == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.enable_data_parallel:
        device = torch.device("cuda")
        model.to(device)
        model = nn.DataParallel(model, device_ids=[0, 1])

    fit_profile(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        num_steps=args.num_steps,
        title=f"baseline_resnet_batch_size_{args.batch_size}_checkpointing_{args.enable_checkpointing}_num_gpus_{args.num_gpus}_dp_{args.enable_data_parallel}",
    )

    # get time stats
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
