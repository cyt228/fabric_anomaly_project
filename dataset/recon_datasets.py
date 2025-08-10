# datasets/recon_paired_dataset.py
import os, glob, cv2, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset

def _read_png_512_rgb(path: str) -> torch.Tensor:
    """
    讀取 PNG，確認大小為 512x512，輸出 [3,512,512]、範圍 [0,1] 的 RGB tensor。
    支援灰階/4通道 PNG，自動轉成 3 通道 RGB。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"讀不到影像：{path}")

    # 通道處理：灰階→RGB；BGRA/BGR→RGB
    if img.ndim == 2:                      # (H,W)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:                # (H,W,4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:                                  # (H,W,3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if (h, w) != (512, 512):
        raise ValueError(f"影像尺寸不是 512x512：{path} -> {(h,w)}")

    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().clamp(0, 1)  # [3,512,512]

class ReconPairedDataset(Dataset):
    """
    兩資料夾成對資料集（專為 PNG + 512x512 + 檔名對應）。
      - normal_dir：正常影像資料夾
      - anom_dir  ：對應的異常影像資料夾
    以「檔名主體」進行一一配對，回傳 (x_anom, x_norm, stem)。
    """
    def __init__(self, normal_dir: str, anom_dir: str):
        normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        anom_paths   = sorted(glob.glob(os.path.join(anom_dir,   "*.png")))
        # print('norm path',normal_paths)
        # print('anorm path',anom_paths)
        if not normal_paths or not anom_paths:
            raise RuntimeError("找不到 PNG 檔，請確認資料夾與副檔名。")

        # 建立 normal 檔名主體 -> 路徑 的對照
        norm_map = {Path(p).stem: p for p in normal_paths}

        # 逐一以檔名主體配對
        pairs = []
        miss_in_normal = []
        for ap in anom_paths:
            stem = Path(ap).stem
            if stem in norm_map:
                pairs.append((ap, norm_map[stem], stem))
            else:
                miss_in_normal.append(Path(ap).name)

        # 嚴格檢查：確保雙方一一對應
        if not pairs:
            raise RuntimeError("沒有任何可配對的影像（檢查兩邊檔名是否一致）。")
        if miss_in_normal:
            raise RuntimeError(f"異常側有 {len(miss_in_normal)} 個檔名在正常側找不到對應：例如 {miss_in_normal[:5]}")

        # 也檢查 normal 是否有多餘
        miss_in_anom = [Path(p).stem for p in normal_paths if Path(p).stem not in {s for _,_,s in pairs}]
        if miss_in_anom:
            raise RuntimeError(f"正常側有 {len(miss_in_anom)} 個檔名在異常側找不到對應：例如 {miss_in_anom[:5]}")

        self.pairs = sorted(pairs, key=lambda x: x[2])  # 依檔名主體排序，確保結果穩定

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ap, npth, stem = self.pairs[i]
        x_anom = _read_png_512_rgb(ap)    # [3,512,512] in [0,1]
        x_norm = _read_png_512_rgb(npth)  # [3,512,512] in [0,1]
        return x_anom, x_norm, stem
