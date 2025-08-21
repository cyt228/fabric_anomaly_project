
import os
import math
import csv
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from torchvision import transforms
    from torchvision.utils import save_image
except Exception as e:
    raise RuntimeError("This script requires torchvision. Please install torchvision.") from e

# -------------------------------
# Utils
# -------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[-1].lower() in IMG_EXTS

def list_images(d: str) -> List[str]:
    if d is None:
        return []
    paths = [os.path.join(d, f) for f in os.listdir(d) if is_image_file(os.path.join(d, f))]
    return sorted(paths)

def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return x.add(1).div(2).clamp(0, 1)

def calc_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target in [0,1]
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1).clamp_min(eps)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr  # shape: [B]

def _gaussian_window_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def _gaussian_kernel(window_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    _1d = _gaussian_window_1d(window_size, sigma).to(device)
    _2d = (_1d[:, None] * _1d[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,w,w]
    kernel = _2d.repeat(channels, 1, 1, 1)  # depthwise conv
    return kernel

@torch.no_grad()
def calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Compute SSIM per image (averaged over channels & pixels).
    pred/target expected in [0,1], shape [B,C,H,W]
    """
    B, C, H, W = pred.shape
    device = pred.device
    K1, K2 = 0.01, 0.03
    L = 1.0  # dynamic range for [0,1]
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    kernel = _gaussian_kernel(window_size, sigma, C, device)
    padding = window_size // 2

    mu1 = F.conv2d(pred, kernel, groups=C, padding=padding)
    mu2 = F.conv2d(target, kernel, groups=C, padding=padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, groups=C, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, groups=C, padding=padding) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, groups=C, padding=padding) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # average over pixels and channels
    ssim_per_img = ssim_map.view(B, C, -1).mean(dim=(1, 2))
    return ssim_per_img

# -------------------------------
# Data
# -------------------------------

class TestPairs(Dataset):
    def __init__(self, input_dir: str, target_dir: str = None, img_size: int = 256):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.inputs = list_images(input_dir)
        self.has_target = False
        self.targets_map = {}
        if target_dir and os.path.isdir(target_dir):
            tgt_list = list_images(target_dir)
            self.targets_map = {os.path.splitext(os.path.basename(p))[0]: p for p in tgt_list}
            # Determine if any targets exist to evaluate
            self.has_target = any(os.path.splitext(os.path.basename(ip))[0] in self.targets_map
                                  for ip in self.inputs)

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        self.tf_target = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        ip_path = self.inputs[idx]
        name = os.path.basename(ip_path)
        name_noext = os.path.splitext(name)[0]

        img = Image.open(ip_path).convert("RGB")
        inp = self.tf(img)

        tgt_path = self.targets_map.get(name_noext, None)
        if tgt_path is not None:
            tgt_img = Image.open(tgt_path).convert("RGB")
            tgt = self.tf_target(tgt_img)  # keep [0,1] for metrics
            has_tgt = True
        else:
            # dummy target
            tgt = torch.zeros(3, inp.shape[1], inp.shape[2])
            has_tgt = False

        return inp, tgt, has_tgt, name

def collate_fn(batch):
    inps, tgts, has_tgts, names = zip(*batch)
    inps = torch.stack(inps, dim=0)
    tgts = torch.stack(tgts, dim=0)
    has_tgts = torch.tensor(has_tgts, dtype=torch.bool)
    return inps, tgts, has_tgts, list(names)

# -------------------------------
# Model loader
# -------------------------------

def build_model(img_channels: int = 3):
    # Import your UNet from model.py
    from unet_model import UNetGenerator
    return UNetGenerator()


def load_checkpoint_into(model: nn.Module, ckpt_path: str, map_location=None):
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict):
        # Try common keys
        for key in ["model_state_dict", "state_dict", "model", "netG", "generator"]:
            if key in obj:
                sd = obj[key]
                if isinstance(sd, nn.Module):
                    sd = sd.state_dict()
                model.load_state_dict(sd, strict=False)
                return
        # Maybe it's already a state dict
        try:
            model.load_state_dict(obj, strict=False)
            return
        except Exception:
            pass
        # Some trainers save under 'ema' etc.
        for k, v in obj.items():
            if isinstance(v, dict):
                try:
                    model.load_state_dict(v, strict=False)
                    return
                except Exception:
                    continue
        raise RuntimeError(f"Could not find a compatible state_dict in checkpoint keys: {list(obj.keys())}")
    else:
        # raw state dict
        model.load_state_dict(obj, strict=False)

# -------------------------------
# Main
# -------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data
    dataset = TestPairs(args.test_input_dir, args.test_target_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate_fn)

    # Model
    netG = build_model(img_channels=3).to(device)
    load_checkpoint_into(netG, args.ckpt, map_location=device)
    netG.eval()

    # Eval CSV
    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["filename", "psnr", "ssim", "l1", "mse"])

    has_any_target = False
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_l1 = 0.0
    sum_mse = 0.0
    count_eval = 0

    autocast_dtype = torch.float16 if (device.type == "cuda" and args.amp) else torch.float32

    with torch.no_grad():
        for batch in loader:
            inps, tgts, has_tgts, names = batch
            inps = inps.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype) if (device.type == "cuda" and args.amp) else torch.cuda.amp.autocast(enabled=False):
                outs = netG(inps)

            # Save recon images
            outs_01 = denorm_to_01(outs.float().detach().cpu())
            for i, name in enumerate(names):
                save_path = os.path.join(args.out_dir, name)
                save_image(outs_01[i], save_path)

            # Evaluate if targets available
            if has_tgts.any():
                has_any_target = True
                tgts_01 = tgts.clamp(0,1).float()
                psnr = calc_psnr(outs_01, tgts_01)
                ssim = calc_ssim(outs_01.to(device), tgts_01.to(device)).cpu()
                l1 = F.l1_loss(outs_01, tgts_01, reduction="none").view(outs_01.size(0), -1).mean(dim=1)
                mse = F.mse_loss(outs_01, tgts_01, reduction="none").view(outs_01.size(0), -1).mean(dim=1)

                for i, name in enumerate(names):
                    if has_tgts[i]:
                        writer.writerow([name,
                                         float(psnr[i].item()),
                                         float(ssim[i].item()),
                                         float(l1[i].item()),
                                         float(mse[i].item())])
                        sum_psnr += float(psnr[i].item())
                        sum_ssim += float(ssim[i].item())
                        sum_l1 += float(l1[i].item())
                        sum_mse += float(mse[i].item())
                        count_eval += 1
                    else:
                        writer.writerow([name, "", "", "", ""])
            else:
                # No targets at all; still write rows without metrics
                for name in names:
                    writer.writerow([name, "", "", "", ""])

    csv_f.close()

    print(f"Reconstructed images saved to: {args.out_dir}")
    if has_any_target and count_eval > 0:
        avg_psnr = sum_psnr / count_eval
        avg_ssim = sum_ssim / count_eval
        avg_l1 = sum_l1 / count_eval
        avg_mse = sum_mse / count_eval
        print("Evaluation on pairs found:")
        print(f"  Images evaluated: {count_eval}")
        print(f"  PSNR (avg): {avg_psnr:.4f} dB")
        print(f"  SSIM (avg): {avg_ssim:.4f}")
        print(f"  L1/MAE (avg): {avg_l1:.6f}")
        print(f"  MSE (avg): {avg_mse:.6f}")
        print(f"Per-image metrics CSV: {csv_path}")
    else:
        print("No matching targets found for evaluation. Only reconstructions were saved.")

def build_argparser():
    p = argparse.ArgumentParser(description="Test (inference + evaluate) with best reconstruction model.")
    p.add_argument("--test_input_dir", type=str, required=True, help="Folder of test defect images (inputs).")
    p.add_argument("--test_target_dir", type=str, default=None, help="Optional folder of ground-truth clean images to evaluate.")
    p.add_argument("--ckpt", type=str, default="checkpoints_recon/recon_best.pt", help="Path to best checkpoint.")
    p.add_argument("--out_dir", type=str, default="recon", help="Output folder to save reconstructed images.")
    p.add_argument("--img_size", type=int, default=512, help="Resize images to this size for inference.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--amp", action="store_true", help="Use CUDA AMP for faster inference.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
