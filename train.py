#train.py

import torch
from tqdm import tqdm
import numpy as np
import os

def train(model, epochs, trainloader, valloader, optimizer, criterion, dir_path, device):
    losses = []
    accuracies = []
    for epoch in range(epochs):
        loss_epoch, acc_epoch = train_batch(model, trainloader, epoch, optimizer, criterion, device)
        val_accuracy = evaluate(model, valloader, device)
        losses.append(loss_epoch)
        accuracies.append(acc_epoch)

        torch.save(losses, os.path.join(dir_path, f"loss_train_unt_epoch{epoch}.pt"))
        torch.save(accuracies, os.path.join(dir_path, f"accuracy_train_epoch{epoch}.pt"))
        torch.save(val_accuracy, os.path.join(dir_path, f"accuracy_val_epoch{epoch}.pt"))

    return model

def train_batch(model, trainloader, epoch, optimizer, criterion, device):
    model.train()
    progress_bar = tqdm(total=len(trainloader), unit='step')
    losses = []
    correct = 0
    total = 0
    for (data, labels) in trainloader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(data)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar.set_postfix(loss=np.mean(losses), accuracy=correct/total)  # Update the loss and accuracy value
        progress_bar.update(1)

    return np.mean(losses), correct / total

def evaluate(model, loader, device):
    model.eval()
    accuracy = []
    correct = 0
    total = 0

    progress_bar = tqdm(total=len(loader), unit='step')

    with torch.no_grad():
        for (data, labels) in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds = preds.detach().cpu()
            labels = labels.cpu()

            accuracy.append((preds == labels).float().sum() / len(labels))

            progress_bar.set_description(f"Evaluation")
            progress_bar.set_postfix(accuracy=correct/total)
            progress_bar.update(1)

    return correct / total
