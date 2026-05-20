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
        "LeakyReLU": nn.LeakyReLU(0.1, inplace=True),
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


def compute_groups(num_features: int,
                   channels_per_group: int = 8) -> int:
    if num_features <= channels_per_group:
        return 1   # Insufficient channels for one group, put all in one group

    ideal_g = num_features // channels_per_group

    # Search downward from ideal_g, find the first value that divides C evenly
    for g in range(ideal_g, 0, -1):
        if num_features % g == 0:
            return g

    return 1   # Fallback


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


# Using activation function before residual connection
class ResnetBlock(nn.Module):
    def __init__(self, nin, nout, act_type="ReLU", norm_layer="Group", dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=3, padding=1),
            normlayer(nout, norm_layer),
            activation(act_type),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # Using Dropout after activation yields best accuracy improvement and fastest convergence

            nn.Conv2d(nout, nout, kernel_size=3, padding=1),
            normlayer(nout, norm_layer),
            activation(act_type),
        )
        self.skip_proj = nn.Conv2d(nin, nout, kernel_size=1) if nin != nout else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_proj(x)


# Using activation function after residual connection
class ResnetBlock1(nn.Module):
    def __init__(self, nin, nout, act_type="ReLU", norm_layer="Group", dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=3, padding=1),
            normlayer(nout, norm_layer),
            activation(act_type),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # Using Dropout after activation yields best accuracy improvement and fastest convergence

            nn.Conv2d(nout, nout, kernel_size=3, padding=1),
            normlayer(nout, norm_layer),
        )
        self.skip_proj = nn.Conv2d(nin, nout, kernel_size=1) if nin != nout else nn.Identity()
        self.act = activation(act_type)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip_proj(x))


# Improved residual block (Pre-activation + Dropout)
class Pre_act_ResnetBlock(nn.Module):
    def __init__(self, nin, nout, act_type="ReLU", norm_layer="Group", dropout=0.0):
        super().__init__()

        # Main branch: norm → act → conv → norm → act → conv
        self.block = nn.Sequential(
            normlayer(nin, norm_layer),
            activation(act_type),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(nin, nout, kernel_size=3, padding=1),

            normlayer(nout, norm_layer),
            activation(act_type),
            nn.Conv2d(nout, nout, kernel_size=3, padding=1),
        )

        self.skip_proj = nn.Conv2d(nin, nout, kernel_size=1) if nin != nout else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip_proj(x)


# Downsampling module
class DownSampling(nn.Module):
    def __init__(self, nin, nout, **kwargs):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Pre_act_ResnetBlock(nin, nout, **kwargs)
        )

    def forward(self, x):
        return self.down(x)


# Upsampling module
class UpSampling(nn.Module):
    def __init__(self, nin, nout, upsample=True, **kwargs):
        super().__init__()
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)  # Upsample only performs spatial upsampling (changes shape but not channels)
            self.conv_block = Pre_act_ResnetBlock(nin + nout, nout, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(nin, nout, kernel_size=4, stride=2, padding=1)
            self.conv_block = ResnetBlock(nout * 2, nout, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        '''
        # Ensure spatial dimensions match (handle odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        '''
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class ResUnet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=16,  # Starting number of channels
                 depth=4,           # Network depth (number of layers)
                 act_type="ReLU",
                 norm_layer=None,
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

        # self.input = DoubleConv(in_channels, channels[0], act_type=act_type, norm_layer=norm_layer, dropout=dropout)
        # self.input = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.input = ResnetBlock1(in_channels, channels[0], act_type=act_type, norm_layer=norm_layer, dropout=dropout)

        # Encoder
        for idx in range(1, depth):
            self.encoder_layers.append(
                DownSampling(channels[idx - 1], channels[idx], act_type=act_type,
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
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1)

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
