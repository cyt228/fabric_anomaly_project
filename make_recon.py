# export_recon_only.py
import os, time, torch, cv2, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset.recon_datasets import ReconPairedDataset
from model.unet import Generator as ReconNet   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_bgr_u8(t):
    x = (t.clamp(0,1).cpu().numpy().transpose(1,2,0)*255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

@torch.no_grad()
def main(normal_dir="dataset/SP3/train/defect-free", anom_dir="dataset/SP3/train/defect_for_classify",
         ckpt="checkpoints/recon_unet_best.pt", out_root="out/recon",
         batch_size=8, num_workers=4):
    os.makedirs(out_root, exist_ok=True)
    ds = ReconPairedDataset(normal_dir, anom_dir)   # 回 (x_anom, x_norm, stem)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    net = ReconNet(input_dim=3, num_filter=64, output_dim=3).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    net.load_state_dict(state["model"]); net.eval()

    t0 = time.time(); n=0
    for x, _, stems in dl:
        x = x.to(DEVICE)                  # [B,3,512,512] in [0,1]
        xr = net(x)                       # 還原圖（正常外觀）
        for i, stem in enumerate(stems):
            cv2.imwrite(str(Path(out_root)/f"{stem}.png"), to_bgr_u8(xr[i]))
            n += 1
    print(f"Saved {n} recon images to {out_root} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()