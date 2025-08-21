#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_unet_restore.py
A minimal but strong U-Net training script for defect -> clean image restoration.

Data layout (paired by filename):
    train/defect/*.png        (input with defects)
    train/defect-free/*.png   (ground-truth clean)

All images are 512x512 PNG and correspond 1:1 by filename.

Usage (example):
    python train_unet_restore.py --data_root /path/to/data --epochs 100 --batch_size 8 --out_dir ./runs/unet_restore

Tip: enable cuDNN benchmarking for fixed-size tensors.
"""

import os
import math
import argparse
from glob import glob
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ----------------------
# Utilities
# ----------------------

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------
# Dataset
# ----------------------

class PairedDefectDataset(Dataset):
    def __init__(self, defect_dir: str, clean_dir: str, augment: bool = True):
        self.defect_dir = defect_dir
        self.clean_dir = clean_dir
        defect_files = sorted(glob(os.path.join(defect_dir, '*.png')))
        clean_files  = sorted(glob(os.path.join(clean_dir,  '*.png')))

        # match by filename intersection
        defect_names = {os.path.basename(p) for p in defect_files}
        clean_names  = {os.path.basename(p) for p in clean_files}
        names = sorted(list(defect_names.intersection(clean_names)))
        if len(names) == 0:
            raise RuntimeError("No paired files found. Ensure matching filenames exist in both folders.")

        self.defect_paths = [os.path.join(defect_dir, n) for n in names]
        self.clean_paths  = [os.path.join(clean_dir,  n) for n in names]

        # transforms
        tfs = [transforms.ToTensor()]
        self.to_tensor = transforms.Compose(tfs)

        # light augmentation that preserves structure
        self.augment = augment
        self.flip_h = transforms.RandomHorizontalFlip(p=0.5)
        self.flip_v = transforms.RandomVerticalFlip(p=0.5)
        self.rot90  = transforms.RandomChoice([
            transforms.RandomRotation(degrees=(0,0)),
            transforms.RandomRotation(degrees=(90,90)),
            transforms.RandomRotation(degrees=(180,180)),
            transforms.RandomRotation(degrees=(270,270)),
        ])

    def __len__(self):
        return len(self.defect_paths)

    def __getitem__(self, idx):
        x_path = self.defect_paths[idx]
        y_path = self.clean_paths[idx]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        # assume 512x512; if not, resize to 512
        if x.size != (512, 512):
            x = x.resize((512, 512), Image.BICUBIC)
        if y.size != (512, 512):
            y = y.resize((512, 512), Image.BICUBIC)

        if self.augment:
            # apply identical spatial transforms to both
            seed = torch.seed()
            torch.manual_seed(seed)
            x = self.flip_h(x)
            torch.manual_seed(seed)
            y = self.flip_h(y)

            seed = torch.seed()
            torch.manual_seed(seed)
            x = self.flip_v(x)
            torch.manual_seed(seed)
            y = self.flip_v(y)

            seed = torch.seed()
            torch.manual_seed(seed)
            x = self.rot90(x)
            torch.manual_seed(seed)
            y = self.rot90(y)

        x = self.to_tensor(x)  # [0,1]
        y = self.to_tensor(y)
        return x, y, os.path.basename(x_path)

# ----------------------
# Model: U-Net (3->3)
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
        # pad if needed
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
# Loss: L1 + (1 - SSIM)
# ----------------------

def gaussian_window(window_size: int, sigma: float, device):
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)],
                         device=device, dtype=torch.float32)
    gauss = gauss / gauss.sum()
    return gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # 2D

def create_window(window_size: int, channel: int, device):
    _2D_window = gaussian_window(window_size, 1.5, device=device).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    # expects [B,C,H,W], values in [0,1]
    device = img1.device
    channel = img1.size(1)
    window = create_window(window_size, channel, device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class MixLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        s = ssim(pred, target)
        # maximize ssim -> minimize (1 - s)
        loss = self.alpha * l1 + (1 - self.alpha) * (1 - s)
        return loss

# ----------------------
# EMA (Exponential Moving Average) of params
# ----------------------

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = UNet()
        self.ema.load_state_dict(model.state_dict())
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

# ----------------------
# Training
# ----------------------

def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    defect_dir = os.path.join(args.data_root, 'train', 'defect')
    clean_dir  = os.path.join(args.data_root, 'train', 'defect-free')
    dataset = PairedDefectDataset(defect_dir, clean_dir, augment=not args.no_augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = MixLoss(alpha=args.alpha)

    ema = ModelEMA(model, decay=0.999)

    ensure_dir(args.out_dir)
    ckpt_path = os.path.join(args.out_dir, 'checkpoint.pt')
    best_path = os.path.join(args.out_dir, 'best_ema.pt')

    # learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)

    global_step = 0
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            # gradient clipping for stability
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            ema.update(model)

            running = 0.9 * running + 0.1 * loss.item() if global_step > 0 else loss.item()
            pbar.set_postfix(loss=f"{running:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            global_step += 1

        scheduler.step()

        # Save checkpoint each epoch
        torch.save({
            'model': model.state_dict(),
            'ema': ema.ema.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch,
            'args': vars(args),
        }, ckpt_path)

        # Evaluate EMA on training data average loss (no val set)
        # We'll compute one pass over a small subset for quick estimate
        model.eval()
        ema.ema.eval()
        with torch.no_grad():
            sample_loss = 0.0
            count = 0
            for i, (x, y, _) in enumerate(loader):
                if i >= max(1, len(loader)//4):  # quick estimate on ~25% of data
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = ema.ema(x)
                l = criterion(pred, y).item()
                sample_loss += l
                count += 1
            sample_loss = sample_loss / max(1, count)

        if sample_loss < best_loss:
            best_loss = sample_loss
            torch.save({'model': ema.ema.state_dict(), 'epoch': epoch, 'args': vars(args)}, best_path)
            print(f"Saved new best EMA model at epoch {epoch} with est. loss {best_loss:.4f} -> {best_path}")

    print("Training complete.")
    print(f"Latest checkpoint: {ckpt_path}")
    print(f"Best EMA checkpoint: {best_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='.', help='Folder containing train/defect and train/defect-free')
    p.add_argument('--out_dir', type=str, default='./runs/unet_restore', help='Where to save checkpoints')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--alpha', type=float, default=0.84, help='weight for L1 in MixLoss')
    p.add_argument('--amp', action='store_true', help='enable mixed precision')
    p.add_argument('--no_augment', action='store_true', help='disable train-time flips/rot90')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ensure_dir(args.out_dir)
    torch.backends.cudnn.benchmark = True
    train(args)
