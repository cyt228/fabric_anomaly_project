import os
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split  # ← 加上 random_split
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

def _can_use_dir(d):
    return d is not None and isinstance(d, str) and len(d) > 0 and os.path.isdir(d)

def build_loaders(args):
    tfm = get_transform(args.img_size)

    # 建立整個訓練資料集
    train_full = PairedImageFolder(args.train_input_dir, args.train_target_dir, transform=tfm)

    # 優先使用使用者提供的獨立驗證資料夾
    if _can_use_dir(args.val_input_dir) and _can_use_dir(args.val_target_dir):
        val_set = PairedImageFolder(args.val_input_dir, args.val_target_dir, transform=tfm)
        train_set = train_full
    else:
        # 沒有驗證資料夾 → 依 val_ratio 從訓練集切分
        n_total = len(train_full)
        if args.val_ratio > 0 and n_total > 1:
            n_val = max(1, int(n_total * args.val_ratio))
            # 確保至少保留 1 張訓練圖
            if n_val >= n_total:
                n_val = n_total - 1
            n_train = n_total - n_val
            g = torch.Generator().manual_seed(args.split_seed)
            train_set, val_set = random_split(train_full, [n_train, n_val], generator=g)
        else:
            train_set, val_set = train_full, None  # 無法切分或 val_ratio=0

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def evaluate_psnr(model, loader, device):
    """對整個資料集計算平均 PSNR；支援 None（回傳 None）。"""
    if loader is None:
        return None
    model.eval()
    total_psnr = 0.0
    total_cnt = 0
    for (inp, tgt, _) in loader:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        out = model(inp)
        # 這裡的 psnr() 以整個 batch 的 MSE 計算一個值；用 batch 大小加權平均
        batch_psnr = psnr(out, tgt)
        total_psnr += float(batch_psnr) * inp.size(0)
        total_cnt += inp.size(0)
    if total_cnt == 0:
        return None
    return total_psnr / total_cnt

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_input_dir', type=str, required=True, help='Directory of defective (input) training images')
    p.add_argument('--train_target_dir', type=str, required=True, help='Directory of clean (target) training images')
    p.add_argument('--val_input_dir', type=str, default=None, help='Directory of defective (input) validation images')
    p.add_argument('--val_target_dir', type=str, default=None, help='Directory of clean (target) validation images')
    p.add_argument('--val_ratio', type=float, default=0.1, help='If no val dirs, split this ratio from train as validation')
    p.add_argument('--split_seed', type=int, default=42, help='Random seed for deterministic split')
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=16)
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

        # 驗證（若無 val，就用訓練集 PSNR 當替代以挑選 best）
        val_psnr = evaluate_psnr(netG, val_loader, device)
        if val_psnr is None:
            # 無法切出驗證集（例如資料太少或 val_ratio=0）
            metric_psnr = evaluate_psnr(netG, train_loader, device)
            metric_name = 'train PSNR'
        else:
            metric_psnr = val_psnr
            metric_name = 'val PSNR'

        # Save last every epoch
        save_ckpt(netG, last_path)

        # Save best by PSNR
        if metric_psnr is not None and metric_psnr > best_psnr:
            best_psnr = metric_psnr
            save_ckpt(netG, best_path)

        shown_val = val_psnr if val_psnr is not None else float('nan')
        print(f'Epoch {epoch:03d}/{args.epochs} | train L1 {avg_l1:.4f} | {metric_name} {metric_psnr:.2f} dB | val PSNR {shown_val:.2f} dB | best {best_psnr:.2f} dB')

    print(f'Done. Saved last to {last_path} and best to {best_path}')

if __name__ == '__main__':
    main()
