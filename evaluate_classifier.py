import os, csv, glob, time, torch, torch.nn as nn, numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
import torchvision.models as tv
from train_classifier import DiffSingleDirDataset, build_model, DEVICE  # 從你剛訓練用的檔案匯入

@torch.no_grad()
def evaluate(orig_dir="dataset/SP3/test/all",
             recon_dir="dataset/SP3/test/recon",
             normals_csv="dataset/SP3/test/all/defect_free_list.csv",
             ckpt="checkpoints_cls/cls_from_singledir_best.pt",
             out_report="eval_test_report.txt",
             out_csv="eval_test_predictions.csv",
             batch_size=32):
    ds = DiffSingleDirDataset(orig_dir, recon_dir, normals_csv, return_stem=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = build_model().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    net.load_state_dict(state["model"]); net.eval()
    loss_fn = nn.CrossEntropyLoss()

    m = defaultdict(float)
    stems_all=[]; y_true=[]; y_pred=[]; prob1_all=[]
    t0 = time.time()
    for x,y,stems in dl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = net(x)
        loss = loss_fn(logits, y)
        m["loss"] += loss.item()*y.size(0)
        pred = logits.argmax(1)
        prob1 = torch.softmax(logits,1)[:,1]
        m["total"] += y.size(0)
        m["correct"] += (pred==y).sum().item()
        m["tp"] += ((pred==1)&(y==1)).sum().item()
        m["tn"] += ((pred==0)&(y==0)).sum().item()
        m["fp"] += ((pred==1)&(y==0)).sum().item()
        m["fn"] += ((pred==0)&(y==1)).sum().item()
        stems_all += list(stems); y_true += y.cpu().tolist()
        y_pred += pred.cpu().tolist(); prob1_all += prob1.cpu().tolist()

    n=max(1,int(m["total"]))
    loss = m["loss"]/n
    acc  = m["correct"]/n
    prec = m["tp"]/max(1,(m["tp"]+m["fp"]))
    rec  = m["tp"]/max(1,(m["tp"]+m["fn"]))
    print(f"TEST: loss={loss:.4f} | acc={acc:.3f} | prec={prec:.3f} | rec={rec:.3f} "
          f"| cm(tp,fp,tn,fn)=({int(m['tp'])},{int(m['fp'])},{int(m['tn'])},{int(m['fn'])}) "
          f"| time={time.time()-t0:.1f}s")

    with open(out_report,"w",encoding="utf-8") as f:
        f.write(f"loss={loss:.6f}\nacc={acc:.6f}\nprecision={prec:.6f}\nrecall={rec:.6f}\n")
        f.write(f"tp={int(m['tp'])}, fp={int(m['fp'])}, tn={int(m['tn'])}, fn={int(m['fn'])}\n")

    import csv as _csv
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=_csv.writer(f); w.writerow(["stem","y_true","y_pred","prob_anom"])
        for s,yt,yp,pr in zip(stems_all,y_true,y_pred,prob1_all):
            w.writerow([s,yt,yp,f"{pr:.6f}"])
    print("Saved:", out_report, "|", out_csv)

if __name__ == "__main__":
    evaluate()
