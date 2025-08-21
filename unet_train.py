import os, math, argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import PairedImageFolder
from unet_model import UNetGenerator

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),   # [-1,1]
    ])

@torch.no_grad()
def eval_psnr(model, loader, device):
    if loader is None: return None
    model.eval()
    tot, cnt = 0.0, 0
    for inp, tgt, _ in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)                 # [-1,1]
        out = (out + 1)/2; tgt = (tgt + 1)/2
        mse = F.mse_loss(out, tgt, reduction="none").view(out.size(0), -1).mean(dim=1).clamp_min(1e-8)
        psnr = 10 * torch.log10(1.0 / mse)
        tot += float(psnr.sum().item()); cnt += out.size(0)
    model.train()
    return tot / max(cnt, 1)

def build_loaders(args):
    tfm = get_transform(args.img_size)
    full = PairedImageFolder(args.train_input_dir, args.train_target_dir, transform=tfm)

    # 若提供 val 資料夾就用；否則從 train 切分
    if args.val_input_dir and args.val_target_dir:
        val = PairedImageFolder(args.val_input_dir, args.val_target_dir, transform=tfm)
        train = full
    else:
        if args.val_ratio > 0 and len(full) > 1:
            n_val = max(1, int(len(full) * args.val_ratio))
            n_val = min(n_val, len(full)-1)
            g = torch.Generator().manual_seed(args.split_seed)
            train, val = random_split(full, [len(full)-n_val, n_val], generator=g)
        else:
            train, val = full, None

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True) if val is not None else None
    return train_loader, val_loader

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_input_dir', type=str, required=True)
    p.add_argument('--train_target_dir', type=str, required=True)
    p.add_argument('--val_input_dir', type=str, default=None)
    p.add_argument('--val_target_dir', type=str, default=None)
    p.add_argument('--val_ratio', type=float, default=0.1)      # 自動切分比例
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--resume', type=str, default='')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = build_loaders(args)

    net = UNetGenerator().to(device)
    if args.resume and os.path.isfile(args.resume):
        net.load_state_dict(torch.load(args.resume, map_location='cpu'), strict=False)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = torch.nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    last_path = os.path.join(args.save_dir, 'recon_last.pt')
    best_path = os.path.join(args.save_dir, 'recon_best.pt')
    best_psnr = -1.0

    for epoch in range(1, args.epochs+1):
        net.train()
        run_l1, n = 0.0, 0
        for inp, tgt, _ in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            out = net(inp)
            loss = l1(out, tgt)
            loss.backward()
            opt.step()
            run_l1 += loss.item() * inp.size(0); n += inp.size(0)
        tr_l1 = run_l1 / max(n, 1)

        # 指標：優先 val PSNR，沒有 val 就用 train PSNR
        val_psnr = eval_psnr(net, val_loader, device)
        metric_psnr = val_psnr if val_psnr is not None else eval_psnr(net, train_loader, device)

        # 存 last / best（raw state_dict，與極簡推論腳本相容）
        torch.save(net.state_dict(), last_path)
        if metric_psnr is not None and metric_psnr > best_psnr:
            best_psnr = metric_psnr
            torch.save(net.state_dict(), best_path)

        print(f"Epoch {epoch:03d}/{args.epochs} | L1 {tr_l1:.4f} | "
              f"{'val' if val_psnr is not None else 'train'} PSNR {metric_psnr:.2f} dB | best {best_psnr:.2f} dB")

    print(f"Done. Saved last to {last_path} and best to {best_path}")

if __name__ == '__main__':
    main()
