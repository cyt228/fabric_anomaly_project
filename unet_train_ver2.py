# unet_train_simple.py (Balanced 感知；epoch=100、無 early stop)
import os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader, random_split

from dataset.recon_datasets import ReconPairedDataset
from model.unet import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- SSIM ----------
from math import exp
def _gaussian(k, s):
    g = torch.tensor([exp(-(x-k//2)**2/(2*s**2)) for x in range(k)])
    return g/g.sum()

def _win(k, c, dev, dt):
    g1 = _gaussian(k, 1.5).to(device=dev, dtype=dt)
    g2 = (g1.unsqueeze(1) @ g1.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return g2.repeat(c,1,1,1)

def ssim(x, y, k=11, eps=1e-8):
    c = x.size(1); dev, dt = x.device, x.dtype; w = _win(k,c,dev,dt)
    mu_x = F.conv2d(x, w, padding=k//2, groups=c); mu_y = F.conv2d(y, w, padding=k//2, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    s_x2 = F.conv2d(x*x, w, padding=k//2, groups=c) - mu_x2
    s_y2 = F.conv2d(y*y, w, padding=k//2, groups=c) - mu_y2
    s_xy = F.conv2d(x*y, w, padding=k//2, groups=c) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_xy + C1)*(2*s_xy + C2))/((mu_x2 + mu_y2 + C1)*(s_x2 + s_y2 + C2) + eps)
    return ssim_map.mean()

# ---------- Balanced Loss ----------
def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y)**2 + eps**2))

def calc_loss(pred, target, metrics, w_l=0.6, w_s=0.4):
    charb = charbonnier_loss(pred, target)
    s     = ssim(pred, target)
    loss  = w_l*charb + w_s*(1 - s)

    bs = pred.size(0)
    metrics['loss']  += loss.item()  * bs
    metrics['charb'] += charb.item() * bs
    metrics['ssim']  += s.item()     * bs
    return loss

def print_metrics(metrics, n, phase):
    avg_loss  = metrics['loss']  / n
    avg_charb = metrics['charb'] / n
    avg_ssim  = metrics['ssim']  / n
    print(f"{phase}: loss={avg_loss:.4f} | Charb={avg_charb:.4f} | SSIM={avg_ssim:.4f}")

# ---------- dataloaders ----------
def make_dataloaders(normal_dir="dataset/SP3/train/defect-free", anom_dir="dataset/SP3/train/defect",
                      batch_size=32, num_workers=4):
    ds = ReconPairedDataset(normal_dir, anom_dir)  # x_anom, x_norm, stem
    n_val = max(1, int(0.1*len(ds))); n_tr = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    dl = {
        'train': DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val'  : DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return dl

# ---------- train ----------
def train_model(model, dataloaders, optimizer, scheduler, num_epochs=100):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_ssim = -1.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n" + "-"*10)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print("LR", optimizer.param_groups[0]['lr'])
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            if phase == 'train':
                scheduler.step()  # CosineAnnealingWarmRestarts：每個 epoch 結束 step 一次

            if phase == 'val':
                epoch_ssim = metrics['ssim'] / epoch_samples
                if epoch_ssim > best_ssim + 1e-4:
                    print("saving best model (by SSIM)")
                    best_ssim = epoch_ssim
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val SSIM: {:.4f}'.format(best_ssim))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    print("device:", device)

    # 1) Model（輸出層 Sigmoid，值域 [0,1]）
    model = Generator(input_dim=3, num_filter=64, output_dim=3).to(device)

    # 2) Optim / Scheduler (Balanced)
    optimizer_ft = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-5
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft, T_0=20, T_mult=2
    )

    # 3) Data
    dataloaders = make_dataloaders(batch_size=32)

    # 4) Train（100 epochs, no early stop）
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=100)

    # 5) Save best
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict()}, "checkpoints/recon_unet_best.pt")
