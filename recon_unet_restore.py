#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recon_unet_restore.py
Run inference (reconstruction) using a trained U-Net checkpoint.

Usage (example):
    python recon_unet_restore.py --ckpt ./runs/unet_restore/best_ema.pt \
        --input_dir /path/to/defect_images \
        --output_dir ./recon_out
"""

import os
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ----------------------
# Model (same as train)
# ----------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*8)
        self.up1 = Up(base*16, base*4)
        self.up2 = Up(base*8, base*2)
        self.up3 = Up(base*4, base)
        self.up4 = Up(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        x = self.outc(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x

# ----------------------
# Inference
# ----------------------

def load_model(ckpt_path, device):
    model = UNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model' in state:
        # training checkpoint with dict
        model.load_state_dict(state['model'])
    else:
        # best_ema.pt saved with {'model': ...}
        model.load_state_dict(state['model'])
    model.eval()
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)

    os.makedirs(args.output_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()

    paths = sorted(glob(os.path.join(args.input_dir, '*.png')))
    if len(paths) == 0:
        raise RuntimeError("No PNGs found in input_dir")

    pbar = tqdm(paths, desc="Reconstructing", ncols=100)
    with torch.no_grad():
        for p in pbar:
            img = Image.open(p).convert('RGB')
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.BICUBIC)
            x = to_tensor(img).unsqueeze(0).to(device)
            pred = model(x)
            pred = pred.squeeze(0).cpu()
            out = transforms.ToPILImage()(pred)
            out.save(os.path.join(args.output_dir, os.path.basename(p)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pt)')
    ap.add_argument('--input_dir', type=str, required=True, help='Dir of defect images (.png)')
    ap.add_argument('--output_dir', type=str, default='./recon_out')
    args = ap.parse_args()
    main(args)
