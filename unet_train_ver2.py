
import os, time, copy
import torch, torch.nn.functional as F
from math import exp
from collections import defaultdict
from torch.utils.data import DataLoader, random_split

from dataset.recon_datasets import ReconPairedDataset     # returns (x_anom, x_norm, stem)
from model.unet import Generator                          # your U-Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- SSIM (expects inputs in [0,1]) ----------------
def _gaussian(k, s):
    g = torch.tensor([exp(-(x-k//2)**2/(2*s**2)) for x in range(k)])
    return g/g.sum()

def _win(k, c, dev, dt):
    g1 = _gaussian(k, 1.5).to(device=dev, dtype=dt)
    g2 = (g1.unsqueeze(1) @ g1.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return g2.repeat(c,1,1,1)

def ssim(x, y, k=5, eps=1e-8):
    c = x.size(1); dev, dt = x.device, x.dtype; w = _win(k,c,dev,dt)
    mu_x = F.conv2d(x, w, padding=k//2, groups=c); mu_y = F.conv2d(y, w, padding=k//2, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    s_x2 = F.conv2d(x*x, w, padding=k//2, groups=c) - mu_x2
    s_y2 = F.conv2d(y*y, w, padding=k//2, groups=c) - mu_y2
    s_xy = F.conv2d(x*y, w, padding=k//2, groups=c) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_xy + C1)*(2*s_xy + C2))/((mu_x2 + mu_y2 + C1)*(s_x2 + s_y2 + C2) + eps)
    return ssim_map.mean()

# ---------------- Loss (L1 + SSIM with warmup) ----------------
def get_weights(epoch, warmup=30, target_ssim=0.2):
    t = min(1.0, epoch / max(1, warmup))    # 0→1
    w_ssim = target_ssim * t                # 前期幾乎純 L1，後期到 L1:SSIM = (1-0.2):0.2
    w_l1   = 1.0 - w_ssim
    return w_l1, w_ssim

def calc_loss(pred, target, metrics, w_l1, w_ssim, k=5):
    l1  = F.l1_loss(pred, target)
    s   = ssim(pred, target, k=k)
    loss = w_l1*l1 + w_ssim*(1 - s)
    bs = pred.size(0)
    metrics['loss'] += loss.item()*bs
    metrics['l1']   += l1.item()*bs
    metrics['ssim'] += s.item()*bs
    return loss

# ---------------- Data ----------------
def make_dataloaders(normal_dir="dataset/SP3/train/defect-free",
                     anom_dir="dataset/SP3/train/defect",
                     batch_size=16, num_workers=4):
    ds = ReconPairedDataset(normal_dir, anom_dir)  # tensors in [0,1]
    n_val = max(1, int(0.1*len(ds))); n_tr = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    return {
        'train': DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val'  : DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }

# ---------------- Train ----------------
def train_model(num_epochs=100, batch_size=16, lr=2e-4, step_size=20, gamma=0.5,
                k_ssim=5, accum_steps=1):

    model = Generator(input_dim=3, num_filter=64, output_dim=3).to(device)
    opt   = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    dls = make_dataloaders(batch_size=batch_size)
    best_wts = copy.deepcopy(model.state_dict()); best_ssim = -1.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n" + "-"*10)
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            metrics = defaultdict(float); nsamp = 0

            opt.zero_grad(set_to_none=True)
            for step, (xa, xr, _) in enumerate(dls[phase]):
                xa, xr = xa.to(device), xr.to(device)

                with torch.set_grad_enabled(phase=='train'):
                    out = model(xa)
                    out = torch.clamp(out, 0.0, 1.0)

                    w_l1, w_ssim = (get_weights(epoch, warmup=30, target_ssim=0.2)
                                    if phase=='train' else (0.8, 0.2))
                    loss = calc_loss(out, xr, metrics, w_l1, w_ssim, k=k_ssim)
                    nsamp += xa.size(0)

                    if phase=='train':
                        (loss/accum_steps).backward()
                        if (step+1) % accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step(); opt.zero_grad(set_to_none=True)

            avg_loss  = metrics['loss']/nsamp
            avg_l1    = metrics['l1']/nsamp
            avg_ssim  = metrics['ssim']/nsamp
            print(f"{phase}: loss={avg_loss:.4f} | L1={avg_l1:.4f} | SSIM={avg_ssim:.4f}")

            if phase=='val' and avg_ssim > best_ssim + 1e-4:
                best_ssim = avg_ssim
                best_wts = copy.deepcopy(model.state_dict())
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({"model": best_wts}, "checkpoints/recon_unet_best.pt")
        sched.step(); print()

    print(f"Best val SSIM: {best_ssim:.4f}")
    model.load_state_dict(best_wts)
    return model

if __name__ == "__main__":
    print("device:", device)
    # 等效 batch 想從 16 提到 32 就設 accum_steps=2
    train_model(num_epochs=100, batch_size=16, lr=2e-4, step_size=20, gamma=0.5,
                k_ssim=5, accum_steps=1)
