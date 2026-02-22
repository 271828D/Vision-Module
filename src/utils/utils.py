import torch as th
import random
from tqdm import tqdm
import numpy as np
import os

# def set_deterministic(seed=42):
#     random.seed(seed)
#     # np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return train_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    with th.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item())
    return val_loss / len(val_loader)


def build_optimizer(model, cfg):
    """Create optimizer."""
    return th.optim.Adam(
        model.parameters(), lr=cfg.training.lr, betas=(0.99, 0.999), eps=1e-8
    )


def save_best_model(val_loss, best_val_loss, model, checkpoint_path):
    """Save model if validation loss improves."""
    if val_loss < best_val_loss:
        print(f"Saving best model to: {checkpoint_path}")
        th.save(model.state_dict(), checkpoint_path)
        return val_loss
    return best_val_loss
