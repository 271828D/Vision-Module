import torch.nn as nn


class ParallelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelConvBlock, self).__init__()
        self.left_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.right_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        left_out = self.left_branch(x)
        right_out = self.right_branch(x)
        return left_out + right_out


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 1, model: str = "cuslin"):
        super(CustomCNN, self).__init__()
        # Initial Conv+BN layers
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Four ParallelConvBlocks
        self.blocks = nn.Sequential(
            ParallelConvBlock(32, 64),
            ParallelConvBlock(64, 128),
            ParallelConvBlock(128, 256),
            ParallelConvBlock(256, 512),
        )
        # New: Conv2d + Global Average Pooling
        self.final_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Avg Pooling
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
