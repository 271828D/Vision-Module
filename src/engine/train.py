from tqdm import tqdm
import torch as th

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())   