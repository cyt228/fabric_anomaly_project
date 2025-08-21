# Custom paired image dataset for defect reconstruction
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _list_images(d):
    return sorted([f for f in os.listdir(d) if os.path.splitext(f.lower())[1] in IMG_EXTS])

class PairedImageFolder(Dataset):
    """
    Expects two directories containing images with matching filenames.
    For example:
      train_input_dir/
         0001.png, 0002.png, ...
      train_target_dir/
         0001.png, 0002.png, ...

    Returns (input_tensor, target_tensor) in range [-1, 1].
    """
    def __init__(self, input_dir, target_dir, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Input or target directory not found: {input_dir}, {target_dir}")

        input_files = set(_list_images(input_dir))
        target_files = set(_list_images(target_dir))
        common = sorted(list(input_files & target_files))
        if not common:
            raise RuntimeError("No matching filenames between input and target directories.")

        self.files = common

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        inp_path = os.path.join(self.input_dir, fname)
        tgt_path = os.path.join(self.target_dir, fname)

        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        if self.transform:
            inp = self.transform(inp)
            tgt = self.transform(tgt)

        return inp, tgt, fname