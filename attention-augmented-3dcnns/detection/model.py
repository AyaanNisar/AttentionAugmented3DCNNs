import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class GlobalAttention3D(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=2):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, D, H, W]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], N=D*H*W
        pos = torch.arange(x_flat.size(1), device=x.device).unsqueeze(0)
        x2, _ = self.attn(x_flat, x_flat, x_flat)
        x2 = self.norm(x2 + x_flat)
        x2 = self.ffn(x2) + x2
        x2 = x2.transpose(1, 2).reshape(B, -1, D, H, W)
        return x2

class NoduleDetector(nn.Module):
    def __init__(self, encoder_weights=None, freeze_encoder=False, attn_dim=128, attn_heads=2):
        super().__init__()
        self.encoder = resnet18(spatial_dims=3, n_input_channels=1)
        if encoder_weights:
            self.encoder.load_state_dict(torch.load(encoder_weights))
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.attn = GlobalAttention3D(512, embed_dim=attn_dim, num_heads=attn_heads)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(attn_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # Forward through encoder up to avgpool
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        # x: [B, 512, D, H, W]
        x = self.attn(x)
        out = self.classifier(x)
        return out 