import torch as th
import pandas as pd
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from src.data.dataloader import get_test_data_loader
from src.models.model import PretrainedModel
import random

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def test(cfg: DictConfig) -> None:
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Set seed and deterministic behavior
    th.manual_seed(cfg.seed)
    th.cuda.manual_seed_all(cfg.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    
    model = PretrainedModel(
        model=cfg.model.model,
        num_classes=cfg.model.num_classes,
        pretrained=False
    ).to(device).eval()

    checkpoint = th.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)

    test_loader = get_test_data_loader(cfg.data)

    results = []
    with th.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for images, labels, paths in progress_bar:
            images = images.to(device)
            outputs = model(images).squeeze()
            scores = th.sigmoid(outputs).cpu().numpy()
            pred_labels = (scores >= 0.5).astype(int)
            if scores.ndim == 0:
                scores = [scores]
                pred_labels = [pred_labels]
            results.extend(zip(paths, scores, pred_labels, labels.cpu().numpy()))
            progress_bar.update(1)

    df = pd.DataFrame(results, columns=["file", "predicted_score", "predicted_label", "true_label"])
    df.to_csv(f"predictions_{cfg.model.model}_{cfg.seed}.csv", index=False)   
    print(f"Saved ../predictions_{cfg.model.model}_{cfg.seed}.csv")

if __name__ == "__main__":
    test()   