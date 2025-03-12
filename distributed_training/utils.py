import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, val_loader, criterion, num_steps=np.inf, rank=0, title=""):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for batch_idx, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()

            val_loss += loss
            val_acc += acc

            if batch_idx >= num_steps:
                break
        assert len(val_loader) == batch_idx + 1 or batch_idx == num_steps
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
    if rank == 0:
        print(f"{title} val_loss: {round(val_loss, 2)}, val_acc: {round(val_acc, 2)}")
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
    log_interval=100,
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
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
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
            if rank == 0 and (batch_idx + 1) % log_interval == 0 and batch_idx > 0:
                train_loss_mean = train_loss / (batch_idx + 1)
                train_acc_mean = train_acc / (batch_idx + 1)
                print(
                    f"Training batch {batch_idx + 1} loss: {round(train_loss_mean, 2)}, acc: {train_acc_mean}"
                )
            num_batches = batch_idx + 1
            if batch_idx >= num_steps:
                break

        train_loss /= num_batches
        train_acc /= num_batches
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
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(train_accs, label="Train acc")
        if val_loader:
            plt.plot(val_accs, label="Val acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.savefig(f"./logs/result_{title}_acc.png", bbox_inches="tight")
        plt.close()
