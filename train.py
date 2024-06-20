#train.py

import torch
from tqdm import tqdm
import numpy as np
import os
import psutil

# Thresholds for memory usage
RAM_THRESHOLD = 30 * 1024 ** 3  # 30 GB
GPU_THRESHOLD = 20 * 1024 ** 3  # 20 GB

def monitor_memory():
    ram_usage = psutil.virtual_memory().used
    gpu_usage = torch.cuda.memory_allocated()
    return ram_usage, gpu_usage

def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch']

def adjust_batch_size(trainloader, factor=0.5):
    batch_size = trainloader.batch_size
    new_batch_size = max(1, int(batch_size * factor))
    trainloader.batch_size = new_batch_size
    print(f"Adjusted batch size to {new_batch_size}")

def train(model, epochs, trainloader, valloader, optimizer, criterion, dir_path, device):
    start_epoch = 0
    checkpoint_path = os.path.join(dir_path, 'checkpoint.pth')

    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resumed training from epoch {start_epoch}")

    losses = []
    accuracies = []
    for epoch in range(start_epoch, epochs):
        loss_epoch, acc_epoch = train_batch(model, trainloader, epoch, optimizer, criterion, device)
        val_accuracy = evaluate(model, valloader, device)
        losses.append(loss_epoch)
        accuracies.append(acc_epoch)

        torch.save(losses, os.path.join(dir_path, f"loss_train_unt_epoch{epoch}.pt"))
        torch.save(accuracies, os.path.join(dir_path, f"accuracy_train_epoch{epoch}.pt"))
        torch.save(val_accuracy, os.path.join(dir_path, f"accuracy_val_epoch{epoch}.pt"))

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Monitor memory usage
        ram_usage, gpu_usage = monitor_memory()
        print(f"RAM usage: {ram_usage / (1024 ** 3):.2f} GB, GPU usage: {gpu_usage / (1024 ** 3):.2f} GB")
        if ram_usage > RAM_THRESHOLD or gpu_usage > GPU_THRESHOLD:
            print(f"Memory usage high: RAM {ram_usage / (1024 ** 3):.2f} GB, GPU {gpu_usage / (1024 ** 3):.2f} GB. Adjusting batch size.")
            adjust_batch_size(trainloader)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(dir_path, 'final_model.pth'))
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
        progress_bar.set_postfix(loss=np.mean(losses), accuracy=correct / total)  # Update the loss and accuracy value
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
            progress_bar.set_postfix(accuracy=correct / total)
            progress_bar.update(1)

    return correct / total
