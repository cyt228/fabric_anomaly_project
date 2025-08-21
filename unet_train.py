import os
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from dataset import PairedImageFolder
from unet_model import UNetGenerator

def psnr(pred, target):
    # inputs in [-1,1]; convert to [0,1] first
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse.item() == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse.item())

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

def build_loaders(args):
    tfm = get_transform(args.img_size)
    train_set = PairedImageFolder(args.train_input_dir, args.train_target_dir, transform=tfm)
    val_set = PairedImageFolder(args.val_input_dir, args.val_target_dir, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_input_dir', type=str, required=True, help='Directory of defective (input) training images')
    p.add_argument('--train_target_dir', type=str, required=True, help='Directory of clean (target) training images')
    p.add_argument('--val_input_dir', type=str, required=True, help='Directory of defective (input) validation images')
    p.add_argument('--val_target_dir', type=str, required=True, help='Directory of clean (target) validation images')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume (optional)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader, val_loader = build_loaders(args)

    netG = UNetGenerator().to(device)
    if args.resume and os.path.isfile(args.resume):
        print(f'Resuming from {args.resume}')
        netG.load_state_dict(torch.load(args.resume, map_location='cpu'))

    optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    l1 = torch.nn.L1Loss()

    best_psnr = -1.0
    last_path = os.path.join(args.save_dir, 'recon_last.pt')
    best_path = os.path.join(args.save_dir, 'recon_best.pt')

    for epoch in range(1, args.epochs+1):
        netG.train()
        total_l1 = 0.0
        for (inp, tgt, _) in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = netG(inp)
                loss = l1(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_l1 += loss.item() * inp.size(0)

        avg_l1 = total_l1 / len(train_loader.dataset)

        # Validation
        netG.eval()
        with torch.no_grad():
            psnr_vals = []
            for (inp, tgt, _) in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                out = netG(inp)
                psnr_vals.append(psnr(out, tgt))
            val_psnr = sum(psnr_vals) / max(1, len(psnr_vals))

        # Save last every epoch
        save_ckpt(netG, last_path)

        # Save best by PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_ckpt(netG, best_path)

        print(f'Epoch {epoch:03d}/{args.epochs} | train L1 {avg_l1:.4f} | val PSNR {val_psnr:.2f} dB | best {best_psnr:.2f} dB')

    print(f'Done. Saved last to {last_path} and best to {best_path}')

if __name__ == '__main__':
    main()