import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, val_loader, criterion, num_steps=np.inf, rank=0, title=""):
    model.eval()
    with torch.no_grad():
        loss = 0
        acc = 0
        for batch_idx, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            preds = outputs.argmax(dim=1, keepdim=True)
            acc += preds.eq(labels.view_as(preds)).sum().item()
            if batch_idx >= num_steps:
                break
        loss /= len(val_loader)
        acc /= len(val_loader)
    if rank == 0:
        print(f"{title} val_loss: {loss}, val_acc: {acc}")
    return loss, acc


def fit(
    model,
    train_loader,
    val_loader=None,
    optimizer=None,
    criterion=None,
    num_epochs=1,
    num_steps=np.inf,
    rank=0,
    title="",
):
    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc.item()
            if rank == 0:
                print(f"Training batch {batch_count}")

            if batch_count >= num_steps:
                break
        train_loss /= batch_count
        train_acc /= batch_count
        if rank == 0:
            print(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if val_loader:
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, num_steps, rank, title=f"epoch: {epoch}"
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    if rank == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train loss")
        if val_loader:
            plt.plot(val_losses, label="Val loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.savefig(f"./logs/result_{title}_loss.png", bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(train_accs, label="Train acc")
        if val_loader:
            plt.plot(val_accs, label="Val acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.savefig(f"./logs/result_{title}_acc.png", bbox_inches="tight")
        plt.show()
