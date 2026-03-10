import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import GroupShuffleSplit
from src.data.dataset import CustomDataset


def get_data_loaders(
    train_csv: str,
    val_csv: str,
    test_size=0.1,
    batch_size=32,
    num_workers=4,
    transform=None,
    random_state=42,
    image_size=224,
):
    train_df = pd.read_csv(train_csv, sep=";")
    val_df = pd.read_csv(val_csv, sep=";")

    transform_data_aug = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),  # Slight rotation for pose variation
            v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Realistic lighting changes
            v2.RandomResizedCrop(
                image_size, scale=(0.8, 1.0)
            ),  # Zoom in/out slightly
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_val = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CustomDataset(train_df, transform=transform_data_aug)
    val_dataset = CustomDataset(val_df, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_data_loaders_per_subject(
    csv_file: str,
    test_size=0.1,
    batch_size=32,
    num_workers=4,
    random_state=42,
    image_size=224,
):

    df = pd.read_csv(csv_file, sep=";")

    # 2. Use GroupShuffleSplit to split by subject (no leakage)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(
        splitter.split(df, groups=df["subject"])
    )  # ← replace 'subject' with your column name

    # 3. Create train and val dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"train shape: {train_df.shape}")
    print(f"training-subjects: {train_df['subject'].nunique()} \n")

    print(f"val shape: {val_df.shape}")
    print(f"val-subjects: {val_df['subject'].nunique()}")

    # 4. Define image transforms
    transform_train = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.7),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_val = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # 5. Create datasets
    train_dataset = CustomDataset(train_df, transform=transform_train)
    val_dataset = CustomDataset(val_df, transform=transform_val)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_data_loader(cfg_data):
    df = pd.read_csv(cfg_data.test_csv, sep=";")  # cfg_data
    transform = v2.Compose(
        [
            v2.Resize((cfg_data.image_size, cfg_data.image_size)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = CustomDataset(df, transform=transform, return_path=True)
    return DataLoader(
        test_dataset,
        batch_size=cfg_data.batch_size,
        shuffle=False,
        num_workers=cfg_data.num_workers,
        pin_memory=True,
    )
