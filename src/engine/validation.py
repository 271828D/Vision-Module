from tqdm import tqdm
import torch as th 

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
    with th.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    return val_loss / len(dataloader)   