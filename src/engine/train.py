import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch as th
import wandb
from datetime import datetime

from src.data.dataloader import get_data_loaders
from src.models.model import PretrainedModel
from src.utils.utils import (
    build_optimizer,
    train_epoch,
    validate,
    save_best_model,
    set_seed,
    EarlyStopping
)


@hydra.main(
    config_path="../../configs", config_name="config", version_base=None
)
def main(cfg: DictConfig) -> None:
    # Setup
    set_seed(cfg.seed)
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    start_time = datetime.now()
    print(f"Starting training at: {start_time.strftime('%Y-%m-%d_%H:%M:%S')}")

    # Data
    train_loader, val_loader = get_data_loaders(**cfg.data)

    # Model
    model = PretrainedModel(
        cfg.model.model,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = th.nn.BCEWithLogitsLoss()

    # Training setup
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    output_dir = HydraConfig.get().runtime.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (
        f"{output_dir}/model_{cfg.model.model}_{cfg.seed}_{timestamp}.pth"
    )

    # Wandb
    safe_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="stress-project", config=safe_cfg)

    # Training loop
    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        # Early stopping & saving
        best_val_loss = save_best_model(
            val_loss, best_val_loss, model, checkpoint_path
        )
        # Check early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            wandb.finish()
            # End time
            end_time = datetime.now()
            print(f"Finished training at: {end_time.strftime('%Y-%m-%d_%H:%M:%S')}")
            print(f"Total training time: {end_time - start_time}")

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

    wandb.finish()

    # End time
    end_time = datetime.now()
    print(f"Finished training at: {end_time.strftime('%Y-%m-%d_%H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")


if __name__ == "__main__":
    main()
