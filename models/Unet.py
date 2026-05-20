# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 2025
Version without attention mechanism
@author: Riveru
"""

import torch
import torch.nn as nn

def activation(act_func="ReLU"):
    """
    Return the specified activation function module.
    Supports string or callable object (e.g., nn.ReLU).
    """
    if callable(act_func):  # Allow direct passing of class or lambda
        return act_func()   # If already callable, return directly

    activations = {
        "ReLU": nn.ReLU(inplace=True),
        "LeakyReLU": nn.LeakyReLU(0.01, inplace=True),
        "ELU": nn.ELU(inplace=True),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU(inplace=True)
    }

    try:
        return activations[act_func]
    except KeyError:
        raise ValueError(
            f"Unsupported activation: '{act_func}'. "
            f"Supported activations: {list(activations.keys())}"
        )


def compute_groups(num_features):
    for num_groups in [16, 8]:
        if num_features % num_groups == 0:
            return num_groups
    for i in range(7, 0, -1):
        if num_features % i == 0:
            return i


def normlayer(num_features, norm_type=None):
    """
    Return the specified normalization layer.
    norm_type ∈ ['Batch', 'Instance', 'Group', None]
    """
    if norm_type is None:
        return nn.Identity()

    norms = {
        "Batch": lambda: nn.BatchNorm2d(num_features),
        "Instance": lambda: nn.InstanceNorm2d(num_features, affine=True),
        "Group": lambda: nn.GroupNorm(
            num_groups=compute_groups(num_features),
            num_channels=num_features,
            affine=True
        )
    }

    try:
        return norms[norm_type]()
    except KeyError:
        raise ValueError(
            f"Unsupported normalization type: '{norm_type}'. "
            f"Supported: {list(norms.keys()) + [None]}"
        )


# U-Net architecture using double convolution as the basic block
class DoubleConv(nn.Module):
    def __init__(self, nin, nout, act_type="ReLU", norm_layer="Group", dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=1, padding=1),  # padding_mode='replicate'
            normlayer(nout, norm_layer),
            activation(act_type),
            nn.Dropout(dropout),  # Using Dropout after activation yields best accuracy improvement and fastest convergence

            nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=3, stride=1, padding=1),  # padding_mode='replicate'
            normlayer(nout, norm_layer),
            activation(act_type),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Downsampling module
class DownSampling(nn.Module):
    def __init__(self, nin, nout, **kwargs):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(nin, nout, **kwargs)
        )

    def forward(self, x):
        return self.down(x)


# Upsampling module
class UpSampling(nn.Module):
    def __init__(self, nin, nout, upsample=True, **kwargs):
        super().__init__()
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsample only performs spatial upsampling (changes shape but not channels)
            self.conv_block = DoubleConv(nin + nout, nout, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(nin, nout, kernel_size=4, stride=2, padding=1)
            self.conv_block = DoubleConv(nout * 2, nout, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure spatial dimensions match (handle odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class Unet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=16,  # Starting number of channels
                 depth=4,           # Network depth (number of layers)
                 act_type="ReLU",
                 norm_layer="Group",
                 upsample=True,
                 dropout=0.0,
                 attention_layers=None):
        super().__init__()
        if attention_layers is None:
            attention_layers = []

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        # Automatically generate channel counts per layer (doubling each time)
        channels = [base_channels * (2 ** i) for i in range(depth)]

        self.input = DoubleConv(in_channels, channels[0], act_type=act_type, norm_layer=norm_layer, dropout=dropout)

        # Encoder
        for idx in range(1, depth):
            in_dim, out_dim = channels[idx - 1], channels[idx]
            self.encoder_layers.append(
                DownSampling(in_dim, out_dim, act_type=act_type,
                             norm_layer=norm_layer, dropout=dropout)
            )

        # Decoder
        for idx in range(depth - 1, 0, -1):
            self.decoder_layers.append(
                UpSampling(channels[idx], channels[idx - 1],
                           upsample=upsample, act_type=act_type,
                           norm_layer=norm_layer, dropout=0.0)
            )

        # Output layer
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        enc_feats = [self.input(x)]

        # Encoding stage
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(enc_feats[-1])
            enc_feats.append(x)

        # Decoding stage
        x = enc_feats.pop()  # Deepest layer
        for layer in self.decoder_layers:
            skip_x = enc_feats.pop()
            x = layer(x, skip_x)

        return self.output(x)


# Test
# model = Unet(in_channels=1, out_channels=1, base_channels=32, depth=5)
# print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
# inp = torch.randn(1, 1, 256, 64)
# out = model(inp)
# summary(model)
