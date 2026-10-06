import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class SimCLR3D(nn.Module):
    def __init__(self, projection_dim=64):
        super().__init__()
        self.backbone = resnet18(spatial_dims=3, n_input_channels=1)
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        # Forward through all layers except the final fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        z = self.projector(feats)
        return z 