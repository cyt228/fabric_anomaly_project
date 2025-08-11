import os, glob, csv, time, random, numpy as np, torch, torch.nn as nn, cv2
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as tv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# CSV：讀「原本無瑕疵」清單(第一欄)
def load_normals_set(csv_path):
    s = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
        if not rows: return s
        start = 1 if rows[0] and ("name" in rows[0][0].lower() or "file" in rows[0][0].lower()) else 0
        for row in rows[start:]:
            if not row: continue
            name = row[0].strip()
            if not name: continue
            stem = os.path.splitext(os.path.basename(name))[0]
            s.add(stem)
    return s

def read_rgb(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED); assert img is not None, path
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy((img.astype(np.float32)/255.0).transpose(2,0,1)).contiguous()
    return x  # [3,H,W] in [0,1]

# Dataset：單一原圖/還原圖資料夾 + CSV 標籤
class DiffSingleDirDataset(Dataset):
    def __init__(self, orig_dir, recon_dir, normals_csv, size=512, return_stem=False, strict=True):
        self.size = size; self.return_stem = return_stem
        self.normals = load_normals_set(normals_csv)

        orig_paths  = {Path(p).stem: p for p in glob.glob(os.path.join(orig_dir,  "*.png"))}
        recon_paths = {Path(p).stem: p for p in glob.glob(os.path.join(recon_dir, "*.png"))}
        common = sorted(set(orig_paths.keys()) & set(recon_paths.keys()))
        if strict and (len(common)==0):
            raise RuntimeError("沒有任何可配對的檔案，請檢查檔名是否一致。")

        # 標籤：在 CSV → 0 (normal)，不在 CSV → 1 (anom)
        self.items = [(orig_paths[k], recon_paths[k], (0 if k in self.normals else 1), k) for k in common]

        # 統計一下 class 分布
        n0 = sum(1 for _,_,y,_ in self.items if y==0)
        n1 = len(self.items)-n0
        print(f"[INFO] pairs={len(self.items)} | normal(label=0)={n0} | anom(label=1)={n1}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        op, rp, y, stem = self.items[i]
        xo = read_rgb(op, self.size)
        xr = read_rgb(rp, self.size)
        xd = (xo - xr).abs()                     # [3,H,W] ∈ [0,1]
        x  = (xd - IMNET_MEAN)/IMNET_STD         # ImageNet normalize
        if self.return_stem:
            return x.float(), torch.tensor(y), stem
        return x.float(), torch.tensor(y)

# ResNet18（2類） 
def build_model():
    m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

# 訓練流程（train/val 兩相 + StepLR）
def calc_metrics(logits, y, metrics):
    loss = metrics["_loss_fn"](logits, y)
    metrics["loss"] += loss.item()*y.size(0)
    pred = logits.argmax(1)
    metrics["total"] += y.size(0)
    metrics["correct"] += (pred==y).sum().item()
    tp = ((pred==1)&(y==1)).sum().item()
    tn = ((pred==0)&(y==0)).sum().item()
    fp = ((pred==1)&(y==0)).sum().item()
    fn = ((pred==0)&(y==1)).sum().item()
    metrics["tp"] += tp; metrics["tn"] += tn; metrics["fp"] += fp; metrics["fn"] += fn
    return loss

def summarize(metrics, phase):
    n=max(1,metrics["total"])
    loss=metrics["loss"]/n
    acc =metrics["correct"]/n
    prec=metrics["tp"]/max(1,(metrics["tp"]+metrics["fp"]))
    rec =metrics["tp"]/max(1,(metrics["tp"]+metrics["fn"]))
    print(f"{phase}: loss={loss:.4f} | acc={acc:.3f} | prec={prec:.3f} | rec={rec:.3f} "
          f"| cm(tp,fp,tn,fn)=({metrics['tp']},{metrics['fp']},{metrics['tn']},{metrics['fn']})")
    return loss

def make_dataloaders(orig_dir, recon_dir, normals_csv, batch_size=16, val_ratio=0.2, num_workers=4, size=512):
    ds = DiffSingleDirDataset(orig_dir, recon_dir, normals_csv, size=size)
    n_val = max(1, int(val_ratio*len(ds))); n_tr = len(ds)-n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))
    dls = {
        "train": DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        "val"  : DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    return dls

def train_val_loop(orig_dir="dataset/SP3/train/defect_for_classify",
                   recon_dir="out/recon",
                   normals_csv="dataset/SP3/train/defect_for_classify/no_defects.csv",
                   out_ckpt="checkpoints_cls/cls_from_singledir_best.pt",
                   size=512, epochs=100, batch_size=16, lr=1e-4, step_size=20, gamma=0.1):
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    dls = make_dataloaders(orig_dir, recon_dir, normals_csv, batch_size=batch_size, size=size)

    net = build_model().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    best=float("inf"); best_w=None
    for ep in range(epochs):
        print(f"Epoch {ep}/{epochs-1}\n" + "-"*10)
        t0 = time.time()

        for phase in ["train", "val"]:
            net.train() if phase=="train" else net.eval()
            metrics = defaultdict(float); metrics["_loss_fn"] = loss_fn

            for x, y in dls[phase]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if phase == "train":
                    opt.zero_grad(set_to_none=True)
                    logits = net(x)
                    loss = calc_metrics(logits, y, metrics)
                    loss.backward(); opt.step()
                else:
                    with torch.no_grad():
                        logits = net(x)
                        _ = calc_metrics(logits, y, metrics)

            if phase == "train":
                sch.step()
                for pg in opt.param_groups:
                    print("LR", pg["lr"])

            val_loss = summarize(metrics, phase)

            # —— 每個 epoch 都先存一份 latest（保底）
            last_path = out_ckpt.replace(".pt", "_last.pt")
            dirpath = os.path.dirname(last_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            torch.save({"model": net.state_dict()}, last_path)

            # —— 只有進步才覆蓋 best
            if phase == "val" and np.isfinite(val_loss) and (val_loss < best - 1e-4):
                best = val_loss
                best_w = net.state_dict().copy()
                best_dir = os.path.dirname(out_ckpt)
                if best_dir: os.makedirs(best_dir, exist_ok=True)
                torch.save({"model": best_w}, out_ckpt)
                print(f"  ↳ saved BEST to {os.path.abspath(out_ckpt)}")

        dt = time.time() - t0
        print(f"{int(dt//60)}m {int(dt%60)}s\n")
    print("Best val loss:", best)

if __name__ == "__main__":
    # 依實際路徑調整
    train_val_loop(
        orig_dir="dataset/SP3/train/defect_for_classify",
        recon_dir="out/recon",
        normals_csv="dataset/SP3/train/defect_for_classify/no_defects.csv", # 「原本無瑕疵」清單
        out_ckpt="checkpoints_cls/cls_from_singledir_best.pt",
        size=512, epochs=100, batch_size=16, lr=1e-4
    )
