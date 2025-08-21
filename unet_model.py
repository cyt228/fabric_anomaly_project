import torch
import torch.nn as nn

# Simple U-Net generator for image-to-image translation (3->3)
# Outputs tanh in [-1, 1]
def _conv_block(in_c, out_c, k=9, s=2, p=1, use_norm=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not use_norm)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def _deconv_block(in_c, out_c, k=9, s=2, p=1, use_norm=True, dropout=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=not use_norm)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        # Encoder
        self.e1 = _conv_block(in_ch, base_ch, use_norm=False)        # 256->128
        self.e2 = _conv_block(base_ch, base_ch*2)                    # 128->64
        self.e3 = _conv_block(base_ch*2, base_ch*4)                  # 64->32
        self.e4 = _conv_block(base_ch*4, base_ch*8)                  # 32->16
        self.e5 = _conv_block(base_ch*8, base_ch*8)                  # 16->8
        self.e6 = _conv_block(base_ch*8, base_ch*8)                  # 8->4
        self.e7 = _conv_block(base_ch*8, base_ch*8)                  # 4->2
        self.e8 = nn.Sequential(                                      # 2->1
            nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.d1 = _deconv_block(base_ch*8, base_ch*8, dropout=True)  # 1->2
        self.d2 = _deconv_block(base_ch*8*2, base_ch*8, dropout=True)# 2->4
        self.d3 = _deconv_block(base_ch*8*2, base_ch*8, dropout=True)# 4->8
        self.d4 = _deconv_block(base_ch*8*2, base_ch*8)              # 8->16
        self.d5 = _deconv_block(base_ch*8*2, base_ch*4)              # 16->32
        self.d6 = _deconv_block(base_ch*4*2, base_ch*2)              # 32->64
        self.d7 = _deconv_block(base_ch*2*2, base_ch)                # 64->128
        self.d8 = nn.ConvTranspose2d(base_ch*2, out_ch, 4, 2, 1)     # 128->256

        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.d8(d7)
        return self.tanh(d8)