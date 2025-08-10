from torch.utils.data import DataLoader
from recon_datasets import ReconPairedDataset

'''
ds = ReconPairedDataset(normal_dir="dataset/SP3/train/defect_free", anom_dir="dataset/SP3/train/defect")
dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

for x_anom, x_norm, stems in dl:
    # 丟進 U-Net：pred = net(x_anom)；loss = L1/SSIM(pred, x_norm)
    pass
'''
ds = ReconPairedDataset("dataset\\SP3\\train\\defect-free", "dataset\\SP3\\train\\defect")
for i in range(3):
    xa, xn, k = ds[i]
    print(k, xa.shape, xn.shape,)