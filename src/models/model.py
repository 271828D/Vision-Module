import torch as th
from torchvision import models

class PretrainedModel(th.nn.Module):
    def __init__(self, model:str, num_classes:int = 1, pretrained:bool = True): 
        super().__init__()

        self.model_name = model
        self.num_classes = num_classes
        self.pretrained = pretrained
        # self.sigmoid = th.nn.Sigmoid()

        if self.model_name == "effv2s":
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            
        elif self.model_name == "effv2l":
            self.backbone = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None)

        elif self.model_name == "effb2":
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)

        elif self.model_name == "effb3":
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)

        elif self.model_name == "mobilev3s":
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)

        elif self.model_name == "mobilev3l":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)

        else:
            raise ValueError(f"model {self.model_name} not supported..")
        
        # Unfreeze all layers 
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        if "efficientnet" in self.backbone.__class__.__name__.lower():
            # EfficientNet uses: classifier[1] as final Linear layer
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = th.nn.Linear(num_ftrs, self.num_classes)
        elif "mobile" in self.backbone.__class__.__name__.lower():
            # MobileNet uses: classifier[3] as final Linear layer
            num_ftrs = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = th.nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        # x = self.backbone(x)
        return self.backbone(x)
    