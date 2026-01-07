import pandas as pd
import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, return_path=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path  # New flag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        image = read_image(str(img_path)) / 255.0
        label = th.tensor(self.df.loc[idx, 'label'], dtype=th.float32)
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label  # Default: keep train.py working
    

    
